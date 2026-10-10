"""Authoritative deterministic game rules, independent of UI and Gymnasium."""
from copy import deepcopy

ACTIONS = ('train', 'hunt', 'ruins', 'forage', 'boss', 'rest', 'attack', 'guard', 'potion', 'flee', 'attackUp', 'defenseUp')
ZONES = {'train': ('훈련장 슬라임',16,3,4), 'hunt': ('폐광의 수호자',18,5,1), 'ruins': ('성채의 파수꾼',18,5,1), 'forage': ('약초 숲의 골렘',14,4,1), 'boss': ('심연의 군주',48,6,0)}

class Game:
    def __init__(self, seed=42): self.reset(seed)
    def reset(self, seed=42):
        self.seed=seed
        self.s=dict(hp=30,maxHp=30,atk=5,defense=0,xp=0,level=1,points=0,weapon=False,armor=False,weaponKills=0,armorKills=0,potions=0,potionKills=0,turn=0,limit=90,enemy=None,done=False,win=False,score=0)
        return self.observe()
    def observe(self): return deepcopy(self.s)
    def attack(self): return self.s['atk']+(4 if self.s['weapon'] else 0)
    def defense(self): return self.s['defense']+(2 if self.s['armor'] else 0)
    def actions(self):
        s=self.s
        if s['done']: return []
        out=['attack','guard'] if s['enemy'] else ['train','hunt','ruins','forage','boss']
        if s['enemy']:
            if s['enemy']['kind']=='boss': out.append('flee')
            if s['potions'] and s['hp']<30: out.append('potion')
        elif s['hp']<30: out.append('rest')
        if s['points']: out+=['attackUp','defenseUp']
        return out
    def step(self,a):
        if a not in self.actions(): raise ValueError(f'Illegal action: {a}')
        s=self.s; before=self.observe(); events=[]; messages=[]; reward=0
        if a in ('attackUp','defenseUp'):
            s['points']-=1
            s['atk' if a=='attackUp' else 'defense']+=2 if a=='attackUp' else 1
            messages.append('공격력 +2' if a=='attackUp' else '방어력 +1')
        else:
            s['turn']+=1
            if a in ZONES:
                name,hp,atk,xp=ZONES[a];s['enemy']=dict(kind=a,name=name,hp=hp,maxHp=hp,atk=atk,xp=xp,phase=0)
                messages.append(name+'와 마주쳤습니다. 진입 1턴.')
            elif a=='rest':
                s['hp']=min(30,s['hp']+6);messages.append('휴식: 체력 +6')
            else:
                e=s['enemy'];heavy=e['kind']=='boss' and e['phase']%3==2
                if a=='attack':
                    damage=self.attack();e['hp']-=damage;messages.append(f'내 공격: {damage} 피해');events.append(dict(type='player_attack',damage=damage))
                elif a=='potion':
                    s['potions']-=1;s['hp']=min(30,s['hp']+14);messages.append('회복약 사용: 체력 +14');events.append(dict(type='potion'))
                elif a=='guard': messages.append('방어 자세: 이번 피해 추가 6 감소');events.append(dict(type='guard'))
                elif a=='flee': messages.append('보스에게서 도주 시도');events.append(dict(type='flee'))
                if e['hp']<=0:
                    messages.append(e['name']+' 격파!');events.append(dict(type='kill',name=e['name'],xp=e['xp']))
                    if e['kind']=='boss':
                        s['done']=s['win']=True;s['score']=100+2*(90-s['turn']);reward=s['score']
                    else:
                        s['xp']+=e['xp']
                        for kind,owned,counter,label in [('hunt','weapon','weaponKills','새벽의 검 · 공격 +4'),('ruins','armor','armorKills','수호자의 갑옷 · 방어 +2')]:
                            if e['kind']==kind and not s[owned]:
                                s[counter]+=1
                                if s[counter]==6: s[owned]=True;messages.append(label+' 획득!');events.append(dict(type='loot',name=label))
                                else: messages.append(f'{label} 획득 진행 {s[counter]}/6')
                        if e['kind']=='forage':
                            s['potionKills']+=1
                            if s['potionKills']==3:
                                s['potionKills']=0;s['potions']+=1;messages.append('회복약 1개 획득!');events.append(dict(type='loot',name='회복약 1개'))
                            else: messages.append(f'회복약 획득 진행 {s["potionKills"]}/3')
                        while s['xp']>=s['level']*4:
                            s['xp']-=s['level']*4;s['level']+=1;s['points']+=1;messages.append(f'레벨 업! Lv.{s["level"]}');events.append(dict(type='level',level=s['level']))
                    s['enemy']=None
                else:
                    damage=max(1,e['atk']+(6 if heavy else 0)-self.defense()-(6 if a=='guard' else 0))
                    s['hp']-=damage
                    if a!='flee':e['phase']+=1
                    messages.append(f'{"강타" if heavy else "몬스터 공격"}: {damage} 피해');events.append(dict(type='enemy_attack',damage=damage,heavy=heavy))
                    if a=='flee' and s['hp']>0: s['enemy']=None;messages.append('도주 성공! 재도전 시 보스 체력 초기화.')
            if not s['done'] and (s['hp']<=0 or s['turn']>=90):
                s['hp']=max(0,s['hp']);s['done']=True;s['score']=-30;reward=-30;messages.append('쓰러졌습니다.' if s['hp']==0 else '시간 초과!')
        return self.observe(),reward,s['done'],dict(messages=messages,events=events,before=before)
