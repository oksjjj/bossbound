import argparse,csv
from pathlib import Path
import numpy as np
from game.env import DungeonEnv
from agents.baseline import Baseline,STRATEGIES
from agents.dqn import DQN

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--episodes',type=int,default=100);p.add_argument('--model');p.add_argument('--output',default='runs/evaluation.csv');a=p.parse_args();Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    names=['random',*STRATEGIES]+(['dqn'] if a.model else []);model=DQN.load(a.model,a.device) if a.model else None;rows=[]
    for name in names:
        results=[]
        for seed in range(a.episodes):
            env=DungeonEnv();obs,info=env.reset(seed=seed);agent=Baseline(name,seed) if name!='dqn' else model;done=False
            while not done:
                action=agent.act(obs,info['action_mask']) if name=='dqn' else agent.act(env.game)
                obs,_,done,_,info=env.step(action)
            results.append(env.game.s)
        wins=[s for s in results if s['win']];row=[name,len(wins)/a.episodes,np.mean([s['turn'] for s in wins]) if wins else '',np.mean([s['score'] for s in results])];rows.append(row);print(f'{name:16} win={row[1]:.1%} winning_turns={row[2]} mean_score={row[3]:.1f}')
    with open(a.output,'w',newline='') as f:writer=csv.writer(f);writer.writerow(['agent','win_rate','winning_turns','mean_score']);writer.writerows(rows)
if __name__=='__main__':main()
