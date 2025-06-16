import torch

if not torch.backends.mps.is_built():
    print("not supported")
else:
    if torch.backends.mps.is_available():
        print("supported")
    else:
        print("issue")
