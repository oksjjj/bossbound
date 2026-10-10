import argparse,csv,json,time
from pathlib import Path
import numpy as np
from game.env import DungeonEnv
from agents.dqn import DQN

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto',choices=['auto','mps','cpu','cuda']);p.add_argument('--episodes',type=int,default=5000);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='runs/dqn');p.add_argument('--warmup',type=int,default=0,help='Optional baseline demonstrations; default pure RL');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);env=DungeonEnv();obs,info=env.reset(seed=args.seed);agent=DQN(len(obs),args.seed,device=args.device);rng=np.random.default_rng(args.seed);replay=[];steps=0;start=time.time();wins=[];print(f"Device: {agent.device}",flush=True)
    (out/'config.json').write_text(json.dumps(vars(args)|{'algorithm':'PyTorch masked DQN','gamma':1.0,'reward_scale':100,'buffer':20000,'batch':64},indent=2))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['episode','win','turn','score','epsilon','loss','policy_source'])
        for ep in range(args.episodes):
            obs,info=env.reset(seed=args.seed+ep);done=False;loss=0;epsilon=max(.05,1-.95*ep/max(1,args.episodes*.8))
            if ep<args.warmup:
                from agents.baseline import Baseline
                teacher=Baseline(['attack','weapon','armor','potion','mixed'][ep%5])
            while not done:
                action=teacher.act(env.game) if ep<args.warmup else agent.act(obs,info['action_mask'],epsilon)
                nxt,reward,done,truncated,ninfo=env.step(action);replay.append((obs,action,reward,nxt,done,ninfo['action_mask']));steps+=1
                if len(replay)>20000:del replay[:1000]
                if len(replay)>=128 and steps%4==0:
                    sample=[replay[i] for i in rng.integers(len(replay),size=64)]
                    batch=tuple(np.array(v) for v in zip(*sample));loss=agent.learn(batch)
                obs,info=nxt,ninfo
            s=env.game.s;wins.append(s['win']);writer.writerow([ep+1,int(s['win']),s['turn'],s['score'],epsilon,loss,'baseline-demo' if ep<args.warmup else 'epsilon-greedy-dqn'])
            if (ep+1)%100==0 or ep+1==args.episodes:
                f.flush();agent.save(out/'latest.pt');print(f'Episode {ep+1} | last100 win {np.mean(wins[-100:]):.1%} | {'baseline-demo' if ep<args.warmup else 'DQN'} | epsilon {epsilon:.2f} | {time.time()-start:.1f}s',flush=True)
    agent.save(out/'final.pt');print(f'Watch: python watch.py --model {out}/final.pt')
if __name__=='__main__':main()
