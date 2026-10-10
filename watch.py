import argparse,json
from game.renderer import Window
from agents.baseline import Baseline,STRATEGIES
if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group();group.add_argument('--model');group.add_argument('--baseline',choices=['random',*STRATEGIES]);group.add_argument('--replay');p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.replay:
        with open(a.replay) as f:r=json.load(f)
        Window(seed=r['seed'],replay=r['history']).run()
    else:
        if a.model:
            from agents.dqn import DQN
            agent=DQN.load(a.model,a.device);print('Device:',agent.device)
        else:agent=Baseline(a.baseline or 'weapon',a.seed)
        window=Window(agent=agent,seed=a.seed)
        if a.model:window.reload_agent=lambda: DQN.load(a.model,a.device)
        window.run()
