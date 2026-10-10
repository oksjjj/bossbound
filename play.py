import argparse
from game.renderer import Window
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);a=p.parse_args();Window(seed=a.seed).run()
