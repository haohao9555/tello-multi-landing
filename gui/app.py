"""English desktop GUI for the Tello/M30 reinforcement learning project."""
import os
import sys
from pathlib import Path
import argparse

# ROS/Gazebo and Ubuntu's Qt bindings are installed in the system package directory.
sys.path.append('/usr/lib/python3/dist-packages')
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QImage, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)
from backend import ROOT, checkpoints, model_info, episode

COLORS = ['#46dcb1', '#64a9ff', '#f5c66c', '#c497ff']
STYLE = '''
QWidget { background:#0c1321; color:#dfe8f5; font-family:"DejaVu Sans"; font-size:12px; }
QFrame#card { background:#131e30; border:1px solid #263449; border-radius:12px; }
QFrame#card QLabel { background:transparent; }
QLabel#muted { color:#8b9bb2; font-size:11px; }
QLabel#title { font-size:25px; font-weight:700; }
QLabel#section { font-size:14px; font-weight:700; }
QLabel#eyebrow { color:#4cddbb; font-size:10px; font-weight:700; }
QComboBox,QSpinBox,QDoubleSpinBox { background:#0c1524; border:1px solid #34445b; border-radius:6px; padding:8px; min-height:19px; }
QComboBox QAbstractItemView { background:#142136; selection-background-color:#26524c; }
QPushButton { background:#1c2b40; border:1px solid #34445b; padding:10px 16px; border-radius:7px; font-weight:600; }
QPushButton:hover { background:#2b4059; }
QPushButton#primary { background:#47ddb7; color:#062b25; border:0; }
QPushButton#primary:disabled, QPushButton:disabled { background:#1c293a; color:#65768c; border-color:#263449; }
QCheckBox { spacing:9px; }
QTableWidget { background:#131e30; border:0; gridline-color:#253348; }
QTableWidget::item { padding:5px; }
QHeaderView::section { background:#1a283d; color:#8b9bb2; border:0; padding:8px; font-size:10px; }
'''

def label(text, kind=None):
    widget = QLabel(text)
    if kind:
        widget.setObjectName(kind)
    return widget


def card(title, subtitle=None):
    frame = QFrame(); frame.setObjectName('card')
    layout = QVBoxLayout(frame); layout.setContentsMargins(18, 16, 18, 16); layout.setSpacing(12)
    layout.addWidget(label(title, 'section'))
    if subtitle:
        text = label(subtitle, 'muted'); text.setWordWrap(True); layout.addWidget(text)
    return frame, layout


class Scene(QWidget):
    def __init__(self):
        super().__init__(); self.frame = None; self.paths = [[] for _ in range(4)]
        self.setMinimumSize(520, 320)
        self.preview = QPixmap(str(ROOT / 'runs/gazebo_multi_test.png'))

    def reset(self):
        self.frame = None; self.paths = [[] for _ in range(4)]; self.update()

    def show_frame(self, frame):
        self.frame = frame
        for i, pos in enumerate(frame['positions']):
            self.paths[i].append(tuple(pos[:2]))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#09111d'))
        f = self.frame
        if f is None:
            if not self.preview.isNull():
                image = self.preview.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                p.drawPixmap((self.width()-image.width())//2,(self.height()-image.height())//2,image)
                p.fillRect(0, self.height()-54, self.width(), 54, QColor(9,17,29,230))
            p.setPen(QColor('#c4d2e4')); p.drawText(20,self.height()-29,'PROJECT GAZEBO CAPTURE  /  Ready to run a new episode')
            p.setPen(QColor('#8b9bb2')); p.drawText(20,self.height()-12,'Choose a checkpoint and select Run episode to begin.')
            return
        if f['image'] is not None:
            a = f['image']; h,w = a.shape[:2]
            image = QImage(a.data,w,h,a.strides[0],QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap((self.width()-pix.width())//2,(self.height()-pix.height())//2,pix)
            return
        extent = f['bounds'] * 1.08; scale = min(self.width(),self.height())/(extent*2)
        def point(x,y): return QPointF(self.width()/2+x*scale,self.height()/2-y*scale)
        p.setPen(QPen(QColor('#1a2a3e'),1))
        for k in range(-5,6):
            v = k*extent/5
            p.drawLine(point(-extent,v),point(extent,v)); p.drawLine(point(v,-extent),point(v,extent))
        for i,target in enumerate(f['targets']):
            center = point(*target); p.setPen(QPen(QColor(COLORS[i]),2)); p.setBrush(QColor('#19333a'))
            p.drawRoundedRect(int(center.x()-12),int(center.y()-12),24,24,4,4)
            p.drawText(QPointF(center.x()-4,center.y()+4),'H')
        for i,pos in enumerate(f['positions']):
            p.setPen(QPen(QColor(COLORS[i]),1.6))
            for a,b in zip(self.paths[i],self.paths[i][1:]): p.drawLine(point(*a),point(*b))
            c=point(*pos[:2]); p.setBrush(QColor(COLORS[i])); p.drawEllipse(c,5,5)
            p.drawLine(c+QPointF(-10,-10),c+QPointF(10,10)); p.drawLine(c+QPointF(-10,10),c+QPointF(10,-10))
            p.drawText(c+QPointF(14,-12),f'{"NESW"[i]}   {pos[2]:.2f} m')
        p.setPen(QColor('#8b9bb2')); p.drawText(18,26,'FAST SIMULATION  /  TOP VIEW  /  metres')
        p.drawText(18,self.height()-16,'N ↑     Coloured trails show actual policy trajectories')


class Worker(QThread):
    frame = pyqtSignal(object)
    failed = pyqtSignal(str)
    def __init__(self, args):
        super().__init__(); self.args=args
    def run(self):
        try:
            for frame in episode(*self.args, stopped=self.isInterruptionRequested): self.frame.emit(frame)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class Window(QMainWindow):
    def __init__(self):
        super().__init__(); self.worker=None; self.config=None
        self.setWindowTitle('AeroLab | Wind-aware Drone Landing'); self.resize(1440,970)
        main=QWidget(); self.setCentralWidget(main); outer=QVBoxLayout(main)
        outer.setContentsMargins(28,22,28,18); outer.setSpacing(18)
        header=QHBoxLayout(); titles=QVBoxLayout()
        titles.addWidget(label('A E R O L A B   /   REINFORCEMENT LEARNING', 'eyebrow'))
        titles.addWidget(label('Wind-aware Drone Landing', 'title'))
        titles.addWidget(label('Configure the environment. Observe the policy. Evaluate the landing.', 'muted'))
        header.addLayout(titles); header.addStretch(); self.badge=label('●  SYSTEM READY'); self.badge.setStyleSheet('color:#47ddb7; padding:12px;')
        header.addWidget(self.badge); outer.addLayout(header)
        body=QHBoxLayout(); body.setSpacing(18); outer.addLayout(body,1)
        left=QVBoxLayout(); left.setSpacing(16); body.addLayout(left)
        vehicle,vl=card('01  Aircraft & policy','Select a trained controller from your project.')
        vehicle.setFixedWidth(330); left.addWidget(vehicle)
        vl.addWidget(label('AIRCRAFT', 'muted')); self.aircraft=QComboBox(); self.aircraft.addItems(['DJI Tello','DJI Matrice 30']); vl.addWidget(self.aircraft)
        vl.addWidget(label('TRAINED CHECKPOINT','muted')); self.models=QComboBox(); vl.addWidget(self.models)
        self.model_note=label('', 'muted'); self.model_note.setWordWrap(True); vl.addWidget(self.model_note)
        env,el=card('02  Environment','Runtime settings apply to this evaluation episode.'); left.addWidget(env)
        self.wind=QDoubleSpinBox(); self.wind.setRange(0,25); self.wind.setSingleStep(.5); self.wind.setSuffix(' m/s'); self.wind.setValue(2)
        self.speed=QDoubleSpinBox(); self.speed.setRange(.1,30); self.speed.setSingleStep(.1); self.speed.setSuffix(' m/s')
        for text, widget in [('MEAN WIND SPEED',self.wind),('MAX HORIZONTAL SPEED',self.speed)]:
            el.addWidget(label(text,'muted')); el.addWidget(widget)
        self.delay_on=QCheckBox('Enable command delay'); self.delay_on.setChecked(True); el.addWidget(self.delay_on)
        self.delay=QSpinBox(); self.delay.setRange(0,2000); self.delay.setSingleStep(100); self.delay.setSuffix(' ms'); self.delay.setValue(100); el.addWidget(self.delay)
        el.addWidget(label('100 ms control steps · rounded to nearest step','muted'))
        el.addWidget(label('RENDERING BACKEND','muted')); self.mode=QComboBox(); self.mode.addItems(['Fast simulation','Gazebo']); el.addWidget(self.mode)
        row=QHBoxLayout(); row.addWidget(label('Episode seed','muted')); self.seed=QSpinBox(); self.seed.setRange(0,999999); row.addWidget(self.seed); el.addLayout(row)
        guard,gl=card('Preflight check'); left.addWidget(guard)
        self.guard=label(''); self.guard.setWordWrap(True); gl.addWidget(self.guard)
        self.run_button=QPushButton('▶  Run episode'); self.run_button.setObjectName('primary'); gl.addWidget(self.run_button)
        self.stop_button=QPushButton('■  Stop'); self.stop_button.setEnabled(False); gl.addWidget(self.stop_button)
        left.addStretch()
        right=QVBoxLayout(); right.setSpacing(16); body.addLayout(right,1)
        view,sv=card('Simulation viewport'); right.addWidget(view,1)
        bar=QHBoxLayout(); self.view_mode=label('PROJECT CAPTURE  ·  NOT LIVE','eyebrow'); bar.addWidget(self.view_mode); bar.addStretch(); self.clock=label('SIM TIME   00.0 s','muted'); bar.addWidget(self.clock); sv.addLayout(bar)
        self.scene=Scene(); sv.addWidget(self.scene,1)
        self.message=label('Ready. Choose your aircraft and environment settings.','muted'); self.message.setWordWrap(True); sv.addWidget(self.message)
        lower=QHBoxLayout(); lower.setSpacing(16); right.addLayout(lower)
        info,il=card('Neural network','Read from the selected checkpoint'); lower.addWidget(info,1)
        self.info=label('Loading model information…'); self.info.setWordWrap(True); self.info.setTextInteractionFlags(Qt.TextSelectableByMouse); il.addWidget(self.info)
        self.source=label('','muted'); self.source.setWordWrap(True); il.addWidget(self.source)
        telemetry,tl=card('Fleet telemetry','Four aircraft · assigned landing platforms'); lower.addWidget(telemetry,1)
        self.table=QTableWidget(4,4); self.table.setHorizontalHeaderLabels(['DRONE','HEIGHT','SPEED','STATUS']); self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.table.setFixedHeight(170); self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        tl.addWidget(self.table)
        for i,name in enumerate(['NORTH','EAST','SOUTH','WEST']):
            for j,text in enumerate([name,'—','—','READY']): self.table.setItem(i,j,QTableWidgetItem(text))
            self.table.item(i,0).setForeground(QColor(COLORS[i]))
        outer.addWidget(label('43008  /  ASSIGNMENT 3     •     Policy evaluation console     •     Simulation only','muted'))
        self.aircraft.currentIndexChanged.connect(self.refresh_models); self.models.currentIndexChanged.connect(self.refresh_info)
        self.wind.valueChanged.connect(self.preflight); self.delay_on.toggled.connect(self.delay.setEnabled)
        self.run_button.clicked.connect(self.start); self.stop_button.clicked.connect(self.stop)
        self.refresh_models()

    def refresh_models(self):
        profile='tello' if self.aircraft.currentIndex()==0 else 'm30'
        self.models.blockSignals(True); self.models.clear()
        for name,path in checkpoints(profile): self.models.addItem(name,str(path))
        preferred = 'fixed_w3_ind' if profile == 'tello' else 'm30_w15_ind'
        index = self.models.findText(preferred)
        if index >= 0: self.models.setCurrentIndex(index)
        self.models.blockSignals(False)
        self.mode.setCurrentIndex(0)
        self.mode.model().item(1).setEnabled(profile=='tello')
        self.scene.reset(); self.refresh_info()

    def refresh_info(self):
        path=self.models.currentData()
        if not path:
            self.config=None; self.info.setText('No compatible checkpoints found.'); self.preflight(); return
        try:
            c,m=model_info(path); self.config=c
            from backend import mt
            p=mt.get_profile(c['env'].get('profile')); self.speed.setValue(c['env'].get('v_max_xy') or p.v_max_xy)
            train=c['training']; wind=c['env'].get('wind_level')
            self.model_note.setText(f'{c["algorithm"].upper()} · Trained wind: {wind if wind is not None else "random"} m/s')
            self.info.setToolTip('\n'.join(m['layers']))
            self.info.setText(f'{c["algorithm"].upper()}  ·  {"GRU" if c["network"]["use_gru"] else "MLP"}  ·  Hidden width {c["network"]["hidden_dim"]}\n'
                f'Actor parameters     {m["actor_params"]:,}\nCritic parameters    {m["critic_params"]:,}\n'
                f'Actor LR  {train["actor_lr"]:g}    Critic LR  {train["critic_lr"]:g}\n'
                f'γ  {train["gamma"]}    GAE λ  {train["gae_lambda"]}    Clip  {train["clip_ratio"]}\n'
                f'Trained steps  {m["steps"]:,}' if isinstance(m['steps'],int) else 'Training steps unavailable')
            self.source.setText('Weights: '+str(Path(path).relative_to(ROOT)))
            self.source.setToolTip('\n'.join(m['layers']))
        except Exception as exc:
            self.config=None; self.info.setText(f'Unable to read checkpoint: {exc}')
        self.preflight()

    def preflight(self):
        blocked=self.wind.value()>15
        self.guard.setText('TAKEOFF BLOCKED\nWind exceeds the project limit of 15 m/s.' if blocked else
            ('Wind within the 15 m/s project limit.\nTello also retains its 3.5 m/s measured-wind check.' if self.aircraft.currentIndex()==0 else
             'Wind within the 15 m/s project limit.\nAircraft measured-wind checks still apply.'))
        self.guard.setStyleSheet('color:'+('#ff8d91' if blocked else '#47ddb7'))
        self.run_button.setEnabled(not blocked and self.config is not None and self.worker is None)

    def start(self):
        if self.wind.value()>15 or self.worker is not None: return
        if self.delay.value()%100:
            self.delay.setValue(100*int(round(self.delay.value()/100)))
        self.scene.reset(); self.message.setText('Loading policy and starting simulation…'); self.badge.setText('●  RUNNING')
        self.view_mode.setText(self.mode.currentText().upper()+'  ·  LIVE EPISODE')
        args=(self.models.currentData(),self.wind.value(),self.speed.value(),self.delay.value() if self.delay_on.isChecked() else 0,self.mode.currentText(),self.seed.value())
        self.worker=Worker(args); self.worker.frame.connect(self.on_frame); self.worker.failed.connect(self.on_error); self.worker.finished.connect(self.finished)
        self.set_inputs(False); self.stop_button.setEnabled(True); self.run_button.setEnabled(False); self.worker.start()

    def set_inputs(self, enabled):
        for widget in [self.aircraft,self.models,self.wind,self.speed,self.delay_on,self.mode,self.seed]: widget.setEnabled(enabled)
        self.delay.setEnabled(enabled and self.delay_on.isChecked())

    def on_frame(self, frame):
        self.scene.show_frame(frame); self.clock.setText(f'SIM TIME   {frame["t"]:04.1f} s')
        for i in range(4):
            speed=sum(float(v)**2 for v in frame['velocities'][i])**.5
            for j,text in enumerate([f'{frame["positions"][i][2]:.2f} m',f'{speed:.2f} m/s',frame['status'][i].replace('_',' ').upper()],1):
                self.table.item(i,j).setText(text)
        if all(s=='no_go' for s in frame['status']): self.message.setText('NO GO · Aircraft preflight measured wind exceeded its configured threshold.')
        elif frame['done']:
            self.message.setText('Episode complete · '+ ' / '.join(s.replace('_',' ') for s in frame['status']))
        else: self.message.setText('Policy running · actual wind '+f'{sum(float(v)**2 for v in frame["wind"])**.5:.2f} m/s')

    def on_error(self, message):
        self.message.setText(message); self.badge.setText('●  ERROR')

    def stop(self):
        if self.worker:
            self.worker.requestInterruption(); self.stop_button.setEnabled(False); self.message.setText('Stopping simulation and releasing resources…')

    def finished(self):
        worker=self.worker; self.worker=None; worker.deleteLater(); self.set_inputs(True); self.stop_button.setEnabled(False)
        self.badge.setText('●  IDLE'); self.preflight()

    def closeEvent(self,event):
        if self.worker:
            self.stop(); self.worker.finished.connect(self.close); event.ignore()
        else: event.accept()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--screenshot',type=Path); parser.add_argument('--smoke-run',action='store_true'); args=parser.parse_args()
    if args.screenshot and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    if (sys.platform.startswith('linux')
            and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
            and os.environ.get('QT_QPA_PLATFORM') not in ('offscreen', 'minimal')):
        parser.exit(2, 'No graphical display is available in this terminal.\n'
                    'Interactive desktop mode requires a graphical desktop or SSH X11 forwarding.\n'
                    'To generate a preview without a desktop, run:\n'
                    '  python gui/app.py '
                    '--screenshot docs/gui_preview.png\n'
                    'Screenshot mode does not open an interactive window.\n')
    app=QApplication(sys.argv[:1]); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    window=Window(); window.show()
    if args.smoke_run: QTimer.singleShot(300,window.start)
    if args.screenshot:
        def capture():
            window.grab().save(str(args.screenshot)); window.close()
        QTimer.singleShot(3500 if args.smoke_run else 600,capture)
    sys.exit(app.exec_())

if __name__=='__main__': main()
