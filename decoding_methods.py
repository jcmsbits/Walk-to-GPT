import torch
import torch.nn.functional as F
a = torch.tensor([1,2,3,4.])
print(F.softmax(a, dim=0))
# tensor([0.0321, 0.0871, 0.2369, 0.6439])
print(F.softmax(a/.5, dim=0))
# tensor([0.0021, 0.0158, 0.1171, 0.8650])
print(F.softmax(a/1.5, dim=0))
# tensor([0.0708, 0.1378, 0.2685, 0.5229])
print(F.softmax(a/1e-6, dim=0))
# tensor([0., 0., 0., 1.])

# Lower temperatures make the model increasingly confident in its top choices, 
# while temperatures greater than 1 decrease confidence. 0 temperature is equivalent 
# to argmax/max likelihood, while infinite temperature corresponds to a uniform sampling.