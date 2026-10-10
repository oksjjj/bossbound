"""Transparent rule baselines; not reinforcement learning."""
from functools import lru_cache
from collections import deque
import random
from game.core import ACTIONS

@lru_cache(maxsize=4096)
def boss_plan(hp,eh,phase,potions,atk,defense):
    q=deque([(hp,eh,phase%3,potions,())]);seen=set()
    while q:
        hp,eh,ph,p,path=q.popleft()
        for a in ['attack','guard','potion']:
            if a=='potion' and (p==0 or hp==30):continue
            nh=hp;ne=eh-atk if a=='attack' else eh;np=p
            if ne<=0:return path+(a,)
            if a=='potion':nh=min(30,nh+14);np-=1
            nh-=max(1,6+(6 if ph==2 else 0)-defense-(6 if a=='guard' else 0))
            if nh<=0:continue
            key=(nh,ne,(ph+1)%3,np)
            if key not in seen:seen.add(key);q.append((*key,path+(a,)))
    return ()

STRATEGIES={'attack':(10,(),0,'attack'), 'mixed':(10,(),0,'mixed'), 'defense':(10,(),0,'defense'), 'weapon':(3,('hunt',),0,'attack'), 'armor':(3,('ruins',),0,'attack'), 'potion':(6,(),1,'attack'), 'weapon-potion':(3,('hunt',),1,'attack'), 'both':(3,('hunt','ruins'),0,'attack'), 'both-potion':(3,('hunt','ruins'),1,'attack')}
class Baseline:
    def __init__(self,name='weapon',seed=42):self.name=name;self.rng=random.Random(seed)
    def act(self,g):
        s=g.s
        if self.name=='random':return ACTIONS.index(self.rng.choice(g.actions()))
        training,gear,pots,mode=STRATEGIES[self.name]
        # Count training XP via cumulative level thresholds; only used before other zones.
        invested=s['level']-1-s['points']
        if s['points']:return ACTIONS.index('defenseUp' if mode=='defense' or (mode=='mixed' and invested%2==1) else 'attackUp')
        e=s['enemy']
        if e:
            if e['kind']=='boss':
                plan=boss_plan(s['hp'],e['hp'],e['phase'],s['potions'],g.attack(),g.defense())
                return ACTIONS.index(plan[0] if plan else 'attack')
            return ACTIONS.index('attack')
        total_xp=2*s['level']*(s['level']-1)+s['xp']
        zone='train' if total_xp<training*4 else next((z for z in gear if not s['weapon' if z=='hunt' else 'armor']),None)
        if zone is None and s['potions']<pots:zone='forage'
        if zone:
            from game.core import ZONES
            _,hp,atk,_=ZONES[zone];cost=((hp+g.attack()-1)//g.attack()-1)*max(1,atk-g.defense())
            return ACTIONS.index('rest' if s['hp']<=cost else zone)
        return ACTIONS.index('rest' if s['hp']<30 else 'boss')
