"""Browser GUI on localhost; expose through the instance's authenticated proxy."""
import argparse
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from backend import (ROOT, checkpoints, model_info, episode, configure_delay,
                     FastMultiTelloEnv, mt, task)

STATIC = Path(__file__).parent / 'web'


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


class Flight:
    def __init__(self):
        self.lock = threading.RLock()
        self.worker = None
        self.stop_event = threading.Event()
        self.command = np.zeros(3)
        self.command_time = 0
        self.selected = 0
        self.camera_mode = "fleet"
        self.camera_drone = 0
        self.generation = 0
        self.state = dict(phase='idle', message='Choose your aircraft and start an episode.', frame=None)
        self.image = b''
        self.catalog = {}
        for profile in ('tello', 'm30'):
            self.catalog[profile] = []
            for i,(name,path) in enumerate(checkpoints(profile)):
                try:
                    c,m = model_info(path)
                    self.catalog[profile].append(dict(id=f'{profile}-{i}', name=name, path=str(path),
                        config=c, model=m, profile=profile))
                except Exception:
                    continue

    def public_catalog(self):
        return {k:[{**v, 'path':str(Path(v['path']).relative_to(ROOT))} for v in values]
                for k,values in self.catalog.items()}

    def validate(self, data):
        profile = data.get('profile')
        if profile not in self.catalog: raise ValueError('Select a valid aircraft.')
        mode, renderer = data.get('mode'),data.get('renderer')
        if mode not in ('manual','ai'): raise ValueError('Invalid control mode.')
        if renderer not in ('fast','gazebo'): raise ValueError('Invalid renderer.')
        if renderer == 'gazebo' and profile != 'tello': raise ValueError('M30 currently supports Fast simulation only.')
        def number(key, low, high):
            value = float(data[key])
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key} must be between {low} and {high}.')
            return value
        wind = number('wind',0,25)
        if wind > 15: raise ValueError('Takeoff blocked: mean wind exceeds 15 m/s.')
        speed = number('speed',.1,30)
        delay = number('delay',0,2000)
        if delay % 100: raise ValueError('Command delay must use 100 ms increments.')
        seed = number('seed',0,999999)
        if seed % 1: raise ValueError('Seed must be an integer.')
        model = next((m for m in self.catalog[profile] if m['id']==data.get('checkpoint')),None)
        if mode == 'ai' and model is None: raise ValueError('Select a compatible checkpoint.')
        return dict(profile=profile, mode=mode, renderer=renderer, wind=wind, speed=speed,
                    delay=delay, seed=int(seed), model=model)

    def start(self, data):
        config = self.validate(data)
        with self.lock:
            if self.worker and self.worker.is_alive(): raise ValueError('Stop the current episode before starting another.')
            self.stop_event = threading.Event(); self.command[:] = 0; self.command_time=0
            self.selected=0; self.generation+=1; self.image=b''
            self.state=dict(phase='starting', message='Starting simulation…', frame=None,
                            mode=config['mode'], renderer=config['renderer'], profile=config['profile'],
                            generation=self.generation)
            self.worker=threading.Thread(target=self.run,args=(config,),daemon=True)
            self.worker.start()

    def set_command(self, data):
        vector = np.asarray(data.get('axes'),dtype=float)
        selected = data.get('selected')
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.abs(vector).max()>1:
            raise ValueError('Axes must contain three finite values between -1 and 1.')
        if type(selected) is not int or selected not in range(4): raise ValueError('Invalid aircraft selection.')
        norm = np.linalg.norm(vector[:2])
        if norm>1: vector[:2]/=norm
        with self.lock:
            if self.state.get('mode')!='manual' or self.state['phase']!='running':
                raise ValueError('Manual control requires a running Manual Flight episode.')
            self.selected=selected; self.command=vector; self.command_time=time.monotonic()

    def commands(self):
        with self.lock:
            action = np.zeros((4,3))
            # Commands expire even if a browser disappears without a release event.
            if time.monotonic()-self.command_time<=.4:
                action[self.selected] = self.command
            else:
                self.command[:]=0
            return action

    def set_camera(self, data):
        mode, drone = data.get('view'), data.get('drone')
        if mode not in ('fleet', 'closeup') or type(drone) is not int or drone not in range(4):
            raise ValueError('Invalid camera view or aircraft.')
        with self.lock:
            self.camera_mode, self.camera_drone = mode, drone

    def camera_target(self):
        with self.lock:
            return self.camera_drone if self.camera_mode == 'closeup' else None

    def stop(self):
        with self.lock:
            self.command[:]=0; self.command_time=0; self.stop_event.set()
            if self.worker and self.worker.is_alive():
                self.state.update(phase='stopping',message='Stopping simulation…')

    def reset(self):
        self.stop()
        worker=self.worker
        if worker: worker.join(timeout=.3)
        with self.lock:
            if worker and worker.is_alive():
                self.state['message']='Stopping simulation. Reset again once it is idle.'
                return
            self.generation+=1; self.image=b''
            self.state=dict(phase='idle',message='Scene reset. Ready for a new episode.',frame=None,generation=self.generation)

    def publish(self, frame):
        image=frame.pop('image',None)
        jpeg=None
        if image is not None:
            buffer=io.BytesIO(); Image.fromarray(image).save(buffer,format='JPEG',quality=82); jpeg=buffer.getvalue()
        with self.lock:
            self.state.update(phase='running',frame=frame,selected=self.selected,
                              command_fresh=time.monotonic()-self.command_time<=.4)
            if jpeg: self.image=jpeg
            if all(s=='no_go' for s in frame['status']):
                self.state['message']='NO GO: aircraft measured-wind check rejected this episode.'
            elif frame['done']: self.state['message']='Episode complete: '+', '.join(frame['status'])
            else: self.state['message']='Manual flight active.' if self.state['mode']=='manual' else 'AI policy running.'

    def run(self, c):
        try:
            if c['mode']=='ai':
                stream=episode(c['model']['path'],c['wind'],c['speed'],c['delay'],
                               'Gazebo' if c['renderer']=='gazebo' else 'Fast simulation',c['seed'],self.stop_event.is_set,
                               camera_target=self.camera_target)
            else: stream=self.manual(c)
            try:
                for frame in stream:
                    if self.stop_event.is_set(): break
                    self.publish(frame)
            finally: stream.close()
            with self.lock:
                self.state['phase']='idle'
                if self.stop_event.is_set(): self.state['message']='Episode stopped. Motors released with simulation resources.'
        except Exception as exc:
            with self.lock: self.state.update(phase='error',message=f'{type(exc).__name__}: {exc}')

    def manual(self,c):
        p=mt.get_profile(c['profile'])
        env=FastMultiTelloEnv(wind_level=c['wind'],preflight=True,v_max_xy=c['speed'],profile=p)
        sim=None
        try:
            _, info=env.reset(seed=c['seed']); configure_delay(env.effects,c['delay'])
            if not info['go']:
                positions=env.pos.copy(); positions[:,2]=0
                yield dict(t=0,positions=positions,velocities=env.vel,status=['no_go']*4,
                    wind=env.wind.current,errors=[None]*4,image=None,targets=p.targets,done=True,bounds=p.bounds_xy)
                return
            if c['renderer']=='gazebo':
                from tello_rl.gazebo_multi import GazeboMultiTello
                from tello_rl.playback import _Episode,FollowCamera
                sim=GazeboMultiTello(camera=True,width=1100,height=650)
                for i in range(4):
                    sim.enable(i,False); sim.set_pose(i,env.pos[i]); sim.enable(i,True)
                sim.step(np.zeros((4,3)))
                rel,vel,_=sim.state(); judge=_Episode(p); camera=FollowCamera()
            for step in range(p.max_steps):
                if self.stop_event.is_set(): return
                started=time.monotonic(); action=self.commands()
                if sim is None:
                    _,_,terminated,truncated,_=env.step(action)
                    rel,vel=env.pos.copy(),env.vel.copy(); image=None
                    status=[o or ('timeout' if truncated else 'flying') for o in env.outcomes]
                    errors=list(env.landing_errors); done=terminated or truncated
                else:
                    cmd=np.zeros((4,3)); was=judge.active.copy(); prev_vz=vel[:,2].copy()
                    for i in np.flatnonzero(was): cmd[i]=env.effects[i].delay(action[i])*env.v_max
                    wind=env.wind.step(task.CONTROL_DT); sim.set_wind(wind); camera.update(sim,rel,was,focus=self.camera_target())
                    sim.step(cmd); rel,vel,image=sim.state()
                    judge.update(rel,vel,prev_vz,was,action,lambda i:sim.enable(i,False))
                    done=not judge.active.any() or step==p.max_steps-1
                    status=[o or ('timeout' if done else 'flying') for o in judge.outcomes]; errors=judge.errors
                yield dict(t=(step+1)*task.CONTROL_DT,positions=rel,velocities=vel,status=status,
                    wind=env.wind.current.copy(),errors=errors,image=image,targets=p.targets,done=done,bounds=p.bounds_xy)
                if done: return
                self.stop_event.wait(max(0,task.CONTROL_DT-(time.monotonic()-started)))
        finally:
            if sim: sim.close()
            env.close()


class Handler(BaseHTTPRequestHandler):
    flight=None
    def reply(self, status, data, content_type='application/json'):
        body = json.dumps(data,default=clean,allow_nan=False).encode() if content_type=='application/json' else data
        self.send_response(status); self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff'); self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError): pass
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/api/catalog': return self.reply(200,self.flight.public_catalog())
        if path=='/api/state':
            with self.flight.lock: return self.reply(200,self.flight.state)
        if path=='/api/frame.jpg':
            with self.flight.lock: data=self.flight.image
            return self.reply(200 if data else 204,data,'image/jpeg')
        routes={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if path in routes:
            file=STATIC/routes[path]; mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}[file.suffix]
            return self.reply(200,file.read_bytes(),mime)
        self.reply(404,{'error':'Not found'})
    def do_POST(self):
        if self.headers.get('X-AeroLab-Request')!='1': return self.reply(403,{'error':'Missing request header'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=8192: raise ValueError('Invalid request size.')
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict): raise ValueError('Expected a JSON object.')
            path=urlsplit(self.path).path
            if path=='/api/start': self.flight.start(data)
            elif path=='/api/command': self.flight.set_command(data)
            elif path=='/api/camera': self.flight.set_camera(data)
            elif path=='/api/stop': self.flight.stop()
            elif path=='/api/reset': self.flight.reset()
            else: return self.reply(404,{'error':'Not found'})
            self.reply(200,{'ok':True})
        except (ValueError,TypeError,KeyError) as exc: self.reply(400,{'error':str(exc)})
    def log_message(self,fmt,*args):
        if '/api/state' not in str(args) and '/api/frame' not in str(args) and '/api/command' not in str(args):
            super().log_message(fmt,*args)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=18080); args=parser.parse_args()
    Handler.flight=Flight(); server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'AeroLab ready: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    finally: Handler.flight.stop(); server.server_close()

if __name__=='__main__': main()
