"""PyTorch DQN: replay, target network, legal action masks, Huber loss."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from game.core import ACTIONS
from game.env import FEATURE_VERSION

def select_device(name='auto'):
    if name=='auto':name='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if name=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable. Check native arm64 Python / PyTorch / macOS, or use --device cpu.')
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    return torch.device(name)

class DQN:
    def __init__(self,nobs,seed=42,hidden=128,device='auto'):
        torch.set_num_threads(1);torch.manual_seed(seed);self.rng=np.random.default_rng(seed);self.device=select_device(device);self.nobs=nobs;self.hidden=hidden
        self.net=nn.Sequential(nn.Linear(nobs,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,len(ACTIONS))).to(self.device)
        self.target=deepcopy(self.net).eval();self.optimizer=torch.optim.Adam(self.net.parameters(),lr=.0005);self.updates=0
    def tensor(self,x,dtype=torch.float32):return torch.as_tensor(x,dtype=dtype,device=self.device)
    def q(self,x):
        with torch.no_grad():return self.net(self.tensor(x)).cpu().numpy()
    def act(self,x,mask,epsilon=0):
        legal=np.flatnonzero(mask)
        if len(legal)==0:raise ValueError('No legal actions')
        if self.rng.random()<epsilon:return int(self.rng.choice(legal))
        return int(np.argmax(np.where(mask,self.q(x),-np.inf)))
    def learn(self,batch,lr=.0005,gamma=1.0):
        x,a,r,nx,done,masks=batch
        x=self.tensor(x);a=self.tensor(a,torch.long);r=self.tensor(r);nx=self.tensor(nx);done=self.tensor(done,torch.bool);mask=self.tensor(masks,torch.bool)
        with torch.no_grad():
            nq=self.target(nx).masked_fill(~mask,-1e9).max(1).values
            target=r+gamma*torch.where(done,torch.zeros_like(nq),nq)
        pred=self.net(x).gather(1,a[:,None]).squeeze(1);loss=nn.functional.smooth_l1_loss(pred,target)
        self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),5);self.optimizer.step();self.updates+=1
        if self.updates%250==0:self.target.load_state_dict(self.net.state_dict())
        return float(loss.detach().cpu())
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        # Atomic replacement allows another process to monitor latest.pt safely.
        temp=Path(str(path)+'.tmp');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in self.net.state_dict().items()},nobs=self.nobs,hidden=self.hidden,feature_version=FEATURE_VERSION,actions=list(ACTIONS)),temp);temp.replace(path)
    @classmethod
    def load(cls,path,device='auto'):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d['feature_version']!=FEATURE_VERSION or tuple(d['actions'])!=ACTIONS:raise ValueError('Model/environment schema mismatch')
        obj=cls(d['nobs'],hidden=d['hidden'],device=device);obj.net.load_state_dict(d['state_dict']);obj.target.load_state_dict(d['state_dict']);return obj
