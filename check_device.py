import platform,sys
import torch
print('Python:',sys.version.split()[0]);print('Architecture:',platform.machine());print('PyTorch:',torch.__version__);print('MPS built:',torch.backends.mps.is_built());print('MPS available:',torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x=torch.ones((8,8),device='mps');print('MPS tensor test:',(x@x)[0,0].item())
else:print('MPS unavailable here; CPU supported.')
