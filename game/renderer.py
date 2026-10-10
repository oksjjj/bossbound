"""Human and AI use identical visual UI. Animation never changes game rules."""
import argparse,json,time
from collections import deque
from pathlib import Path
from copy import deepcopy
import pygame
from .core import Game,ACTIONS

LABELS={'train':('훈련장','HP 16 / 공격 3 / XP 4'),'hunt':('폐광','HP 18 / 공격 5 / 무기 +4'),'ruins':('폐성채','HP 18 / 공격 5 / 방어구 +2'),'forage':('약초 숲','HP 14 / 공격 4 / 회복약'),'boss':('보스 도전','HP 48 / 공격 6 / 3번째 강타'),'rest':('휴식','체력 +6 / 1턴'),'attack':('공격하기','공격 후 자동 반격'),'guard':('방어하기','공격 대신 피해 6 감소'),'potion':('회복약 사용','체력 +14 / 자동 반격'),'flee':('도망가기','보스 공격 1회 후 복귀'),'attackUp':('공격력 투자','포인트 1 / 공격 +2'),'defenseUp':('방어력 투자','포인트 1 / 방어 +1')}
BG=(16,23,19);PANEL=(28,39,31);GREEN=(180,208,140);TEXT=(230,236,224);MUTED=(151,170,145)

class Window:
    def __init__(self,agent=None,seed=42,replay=None):
        pygame.init();self.screen=pygame.display.set_mode((1120,820));pygame.display.set_caption('BOSSBOUND | Python RL');self.clock=pygame.time.Clock()
        # macOS Apple SD Gothic Neo; Linux Noto fallback; custom font supported.
        import os
        font=os.environ.get('BOSSBOUND_FONT') or str(Path(__file__).resolve().parent.parent/'assets'/'NotoSansKR.ttf')
        def load(n):return pygame.font.Font(font,n) if font and Path(font).exists() else pygame.font.SysFont('applesdgothicneo,applegothic,notosanscjkkr,notosanscjk,malgungothic,dejavusans',n)
        self.font=load(19);self.small=load(15);self.title=load(30)
        for f in (self.font,self.small,self.title):f.set_bold(True)
        self.agent=agent;self.seed=seed;self.game=Game(seed);self.logs=deque(maxlen=11);self.frames=deque();self.until=0;self.view=None;self.message='준비가 끝났다면 보스에게 도전하세요';self.pause=agent is not None;self.speed=1;self.last=0;self.history=[];self.buttons=[];self.replay=replay;self.replay_index=0;self.reload_agent=None
    def text(self,t,pos,color=TEXT,font=None):self.screen.blit((font or self.font).render(str(t),True,color),pos)
    def rect(self,r,color=PANEL):pygame.draw.rect(self.screen,color,r,border_radius=10)
    def button(self,key,label,r,enabled=True):
        self.rect(r,(46,68,48) if enabled else (35,43,36));self.text(label,(r[0]+12,r[1]+12),TEXT if enabled else MUTED,self.small);self.buttons.append((pygame.Rect(r),key,enabled))
    def draw(self):
        self.buttons=[];s=self.view or self.game.s;e=s['enemy'];busy=bool(self.frames or self.view)
        self.screen.fill(BG);self.text('BOSSBOUND',(30,22),GREEN,self.title);self.text('사람 플레이' if self.agent is None and self.replay is None else 'AI 관전' if self.replay is None else '기록 다시 보기',(270,35),MUTED)
        self.button('reset','새 도전 / R',(870,20,210,45),not busy)
        self.text(f"TURN {s['turn']} / 90    점수 {s['score']}",(30,80),GREEN)
        atk=s['atk']+(4 if s['weapon'] else 0);df=s['defense']+(2 if s['armor'] else 0)
        self.rect((30,115,670,330));self.text(f"Lv.{s['level']}   HP {s['hp']}/30   공격 {atk}   방어 {df}",(50,132))
        self.text(f"XP {s['xp']}/{s['level']*4}   투자 포인트 {s['points']}   회복약 {s['potions']}개",(50,164),MUTED)
        self.text(('새벽의 검 +4' if s['weapon'] else '기본 검')+' / '+('수호자의 갑옷 +2' if s['armor'] else '방어구 없음'),(50,197),GREEN,self.small)
        pygame.draw.circle(self.screen,(122,164,102),(150,300),40);self.text('탐험자',(120,353))
        pygame.draw.circle(self.screen,(184,104,78) if e else (76,88,69),(570,300),40)
        self.text(e['name'] if e else '야영지',(470,350),TEXT,self.small)
        if e:self.text(f"HP {max(0,e['hp'])}/{e['maxHp']} · 다음 {'강타' if e['kind']=='boss' and e['phase']%3==2 else '공격'}",(445,380),GREEN,self.small)
        self.text(self.message[:44],(50,411),GREEN,self.small)
        if '내 공격' in self.message:pygame.draw.line(self.screen,GREEN,(240,292),(440,292),4);pygame.draw.polygon(self.screen,GREEN,[(440,292),(425,281),(425,303)])
        elif '몬스터 공격' in self.message or '강타:' in self.message:pygame.draw.line(self.screen,(230,153,111),(440,292),(240,292),4);pygame.draw.polygon(self.screen,(230,153,111),[(240,292),(255,281),(255,303)])
        self.rect((725,115,355,330));self.text('모험 기록',(744,131),GREEN)
        for i,t in enumerate(self.logs):self.text(t[:30],(742,170+i*23),TEXT,self.small)
        legal=self.game.actions();playable=self.agent is None and self.replay is None and not busy
        normal=[a for a in legal if a not in ('attackUp','defenseUp')]
        for i,a in enumerate(normal):
            label,desc=LABELS[a];progress=''
            if a in ('hunt','ruins'):progress=f" {s['weaponKills' if a=='hunt' else 'armorKills']}/6"
            if a=='forage':progress=f" {s['potionKills']}/3"
            r=(30+(i%2)*340,465+(i//2)*62,325,54);self.button(a,label+progress,r,playable);self.text(desc,(r[0]+12,r[1]+33),MUTED,self.small)
        self.text('능력치 투자 · 언제든 사용',(725,468),GREEN,self.small)
        self.button('attackUp','공격 +2',(725,500,170,45),playable and 'attackUp' in legal);self.button('defenseUp','방어 +1',(910,500,170,45),playable and 'defenseUp' in legal)
        self.text('장비는 각 사냥터 6회 처치로 확정 획득',(725,565),MUTED,self.small);self.text('회복약은 약초 숲 3회 처치마다 1개',(725,591),MUTED,self.small);self.text('회복약 보유 제한 없음 · 시작 0개',(725,617),MUTED,self.small)
        if self.agent is not None or self.replay is not None:
            self.button('pause','재생' if self.pause else '일시정지',(30,725,140,45));self.button('step','한 행동 / N',(185,725,155,45),not busy);self.button('speed',f'속도 {self.speed}x',(355,725,145,45));self.text('Space: 재생/정지 · N: 한 행동',(520,740),MUTED,self.small)
        self.button('save','기록 저장',(910,725,170,45),not busy)
        if s['done']:self.text('보스 격파 성공!' if s['win'] else '도전 실패',(30,785),GREEN)
        pygame.display.flip()
    def perform(self,action):
        before=self.game.observe();after,reward,done,info=self.game.step(action);self.history.append(dict(action=action,state=after,reward=reward))
        mid=deepcopy(before);mid['turn']=after['turn'];frames=[]
        for ev in info['events']:
            typ=ev['type']
            if typ=='player_attack':mid['enemy']['hp']=max(0,mid['enemy']['hp']-ev['damage']);msg=f"내 공격: {ev['damage']} 피해"
            elif typ=='enemy_attack':mid['hp']=max(0,mid['hp']-ev['damage']);msg=f"{'강타' if ev['heavy'] else '몬스터 공격'}: {ev['damage']} 피해"
            elif typ=='potion':mid['hp']=min(30,mid['hp']+14);mid['potions']-=1;msg='회복약 사용: 체력 +14'
            elif typ=='guard':msg='방어 자세를 취합니다'
            elif typ=='flee':msg='도주를 시도합니다'
            elif typ=='kill':msg=ev['name']+' 격파!'
            elif typ=='loot':msg=ev['name']+' 획득!'
            else:msg=f"레벨 업! Lv.{ev['level']}"
            frames.append((deepcopy(mid),msg))
        frames.append((after,' / '.join(info['messages'][-2:])))
        self.frames.extend(frames)
        for msg in info['messages']:self.logs.appendleft(f"T{after['turn']} {msg}")
    def next_ai(self):
        if self.game.s['done']:return
        if self.replay is not None:
            if self.replay_index>=len(self.replay):return
            a=self.replay[self.replay_index]['action'];self.replay_index+=1
        else:
            from .env import features
            import numpy as np
            if hasattr(self.agent,'q'):a=ACTIONS[self.agent.act(features(self.game),np.array([a in self.game.actions() for a in ACTIONS]))]
            else:a=ACTIONS[self.agent.act(self.game)]
        self.perform(a)
    def save(self):
        Path('runs').mkdir(exist_ok=True);p=Path('runs')/f'replay-{int(time.time())}.json';p.write_text(json.dumps(dict(seed=self.seed,history=self.history),ensure_ascii=False,indent=2));self.logs.appendleft(f'저장: {p.name}')
    def handle(self,key):
        busy=bool(self.frames or self.view)
        if key=='pause':self.pause=not self.pause
        elif key=='speed':self.speed={1:2,2:4,4:8,8:1}[self.speed]
        elif key=='step' and not busy:self.pause=True;self.next_ai()
        elif key=='reset' and not busy:
            if self.reload_agent is not None:self.agent=self.reload_agent()
            self.game.reset(self.seed);self.logs.clear();self.history=[];self.replay_index=0;self.message='준비가 끝났다면 보스에게 도전하세요'
        elif key=='save' and not busy:self.save()
        elif key in self.game.actions() and not busy and self.agent is None and self.replay is None:self.perform(key)
    def run(self):
        running=True
        while running:
            now=time.monotonic()
            for event in pygame.event.get():
                if event.type==pygame.QUIT:running=False
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:running=False
                    elif event.key==pygame.K_r:self.handle('reset')
                    elif (self.agent is not None or self.replay is not None) and event.key in (pygame.K_SPACE,pygame.K_n):self.handle('pause' if event.key==pygame.K_SPACE else 'step')
                elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    for rect,key,enabled in self.buttons:
                        if enabled and rect.collidepoint(event.pos):self.handle(key);break
            if now>=self.until:
                if self.frames:self.view,self.message=self.frames.popleft();self.until=now+.65/self.speed
                elif self.view is not None:self.view=None;self.last=now
                elif not self.pause and (self.agent is not None or self.replay is not None) and now-self.last>.25/self.speed:self.next_ai();self.last=now
            self.draw();self.clock.tick(60)
        pygame.quit()
