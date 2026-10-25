"""Verify local-value isolation, matching actor, update and checkpoint loading."""
import copy, dataclasses, json, math, sys, tempfile
from pathlib import Path
import torch
from tello_rl.train_marl import build_config
from tello_rl.marl_env import VectorTelloMARL
from tello_rl import multi_task as mt
from tello_rl.evaluate_marl import load_policy
from algorithms.ippo import IPPO
from algorithms.mappo import MAPPO
from runner import Runner

torch.set_num_threads(1)
for profile_name in ('tello','m30'):
    cfg=build_config('ippo',1,'cpu',40,2,[])
    cfg['env'].update(profile=profile_name,action_history=mt.PROFILES[profile_name].action_history,
        profile_overrides={'max_steps':10},reward='individual',v_max_xy=5. if profile_name=='m30' else None,wind_level=0.,episode_limit=10)
    cfg['evaluation']['episodes']=2
    env=VectorTelloMARL(2,1,wind_level=0,individual_reward=True,v_max_xy=cfg['env']['v_max_xy'],profile=dataclasses.replace(mt.PROFILES[profile_name],max_steps=10))
    s=env.spec
    info={k:getattr(s,k) for k in ('n_agents','obs_dim','state_dim','n_actions','observation_mask','action_mask')}
    torch.manual_seed(1); algo=IPPO(info,cfg)
    torch.manual_seed(1); mappo=MAPPO(info,cfg)
    assert all(torch.equal(v,mappo.actor.state_dict()[k]) for k,v in algo.actor.state_dict().items())
    runner=Runner(cfg,algo,env); batch=runner.collect(reset_seed=42)
    altered=copy.deepcopy(batch); altered.states.fill_(float('nan'))
    a,b=copy.deepcopy(algo),copy.deepcopy(algo)
    ga=a.critic.gae(batch,batch.rewards,False); gb=b.critic.gae(altered,altered.rewards,False)
    assert all(torch.equal(x,y) for x,y in zip(ga,gb)), 'privileged state leakage'
    ma=a.update(batch); mb=b.update(altered) if False else None
    # Full update must also ignore a poisoned global state; reset RNG for identical minibatches.
    a,b=copy.deepcopy(algo),copy.deepcopy(algo)
    torch.manual_seed(77); ma=a.update(batch)
    torch.manual_seed(77); mb=b.update(altered)
    assert ma==mb and all(math.isfinite(v) for v in ma.values())
    assert all(torch.equal(v,b.actor.state_dict()[k]) for k,v in a.actor.state_dict().items())
    assert a.normalizers.state.count.item()==algo.normalizers.state.count.item()
    assert a.normalizers.obs.count.item()>1
    observations=batch.observations.clone()
    baseline=algo.critic.network(observations)
    observations[:,:,1]+=100
    assert torch.equal(baseline[:,:,0],algo.critic.network(observations)[:,:,0])
    with tempfile.TemporaryDirectory(prefix='ippo-check-') as tmp:
        path=Path(tmp)/'latest.pt'; runner.algorithm=a; runner.save(path)
        restored=IPPO(info,cfg); restored.load_checkpoint_state(torch.load(path,weights_only=False)['algorithm_state'],True)
        assert restored.total_steps==a.total_steps
        assert all(torch.equal(v,restored.normalizers.state_dict()[k]) for k,v in a.normalizers.state_dict().items())
        policy,_=load_policy(path)
        flat=env.envs[0].reset(seed=42)[0] if hasattr(env,'envs') else None
        if flat is not None:
            actions=policy(flat)
            assert actions.shape==(12,) and abs(actions).max()<=1
    print(json.dumps({'profile':profile_name,'observation_dim':s.obs_dim,'env_steps':a.total_steps,'finite_metrics':ma,'state_isolation':'passed','checkpoint':'passed'}))
