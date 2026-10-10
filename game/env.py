"""Gymnasium adapter. The 90-turn horizon is part of the task, hence terminated."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .core import Game,ACTIONS,ZONES

FEATURE_VERSION=1

def features(g):
    s=g.s;e=s['enemy'];kind=e['kind'] if e else 'camp'
    vals=[s['hp']/30,g.attack()/40,g.defense()/20,s['level']/20,s['xp']/80,s['points']/20,s['turn']/90,(90-s['turn'])/90,float(s['weapon']),float(s['armor']),s['potions']/30,s['weaponKills']/6,s['armorKills']/6,s['potionKills']/3]
    vals += [float(kind==k) for k in ['camp',*ZONES]]
    vals += [e['hp']/48 if e else 0,e['atk']/12 if e else 0,e['xp']/4 if e else 0]
    vals += [float(e is not None and e['phase']%3==p) for p in range(3)]
    return np.asarray(vals,dtype=np.float32)

class DungeonEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self):
        self.game=Game();self.action_space=spaces.Discrete(len(ACTIONS));self.observation_space=spaces.Box(0,1,shape=features(self.game).shape,dtype=np.float32)
    def action_masks(self): return np.array([a in self.game.actions() for a in ACTIONS],dtype=bool)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.game.reset(42 if seed is None else seed)
        return features(self.game),self.info()
    def info(self): return dict(action_mask=self.action_masks(),state=self.game.observe())
    def step(self,action):
        if not self.action_space.contains(action): raise ValueError('Action outside Discrete space')
        if ACTIONS[int(action)] not in self.game.actions():
            return features(self.game),0.0,self.game.s['done'],False,self.info()|{'invalid':True,'messages':['현재 사용할 수 없는 행동입니다.']}
        _,reward,done,info=self.game.step(ACTIONS[int(action)]);info.update(self.info())
        # Reward normalization preserves optimal undiscounted score ordering.
        return features(self.game),reward/100,done,False,info
