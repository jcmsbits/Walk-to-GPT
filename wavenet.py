import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
#%matplotlib inline

# read in all the words
words = open("./names.txt", 'r').read().splitlines()
print(len(words))
print(max([len(w) for w in words]))
print(words[:8])


# build the vocabulary of characters and mappings to/from integers
chr = sorted(list(set(''.join(words))))
print(chr)
stoi = {s:i+1 for i,s in enumerate(chr)}
stoi["."] = 0
print(f"stoi: {stoi}")
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)
print(f"itos: {itos}")

# shuffle up the words
import random
random.seed(42)
random.shuffle(words)


# build the dataset
block_size = 8
def build_dataset(words):
    X, Y = [], []    

    for word in words:
        context = [0] * block_size
        for w in word + ".":
            ix = stoi[w]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

for x,y in zip(Xtr[:20], Ytr[:20]):
    print(''.join([itos[ix.item()] for ix in x]), "--->", itos[y.item()])


# Near copy paste of the layers we have developed in Part 3
# -------------------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 #:kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum udpate')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim = True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

# -----------------------------------------------------------------------------------------------------
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
        
    def __call__(self, IX):
        self.out = self.weight[IX]
        print(self.out.shape)
        return self.out
    
    def parameters(self):
        return [self.weight]

# class Flatten:
#     def __call__(self, x):
#         self.out = x.view(x.shape[0], -1)
#         return self.out
    
#     def parameters(self):
#             return []

class FlattenConsecutive:
    def __init__(self, n):
        # n is how many characters need to predict other(block_size)
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)        
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
            return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        # get parameters of all layers, and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]

torch.manual_seed(42); # seed rng for reproducibility

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

C = torch.randn((vocab_size, n_embd))
model = Sequential([ Embedding(vocab_size, n_embd), FlattenConsecutive(2),
    Linear(n_embd * 2, n_hidden, bias = False),
    BatchNorm1d(n_hidden),Tanh(), FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias = False),
    BatchNorm1d(n_hidden),Tanh(), FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias = False),
    BatchNorm1d(n_hidden),Tanh(),Linear(n_hidden, vocab_size)
])

# parameter init
with torch.no_grad():
    model.layers[-1].weight *= 0.1 # the last layer make less confident

# parameters = [C] + [p for layer in layers for p in layer.parameters()]
parameters = model.parameters()
print(([p.nelement() for p in parameters]))
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True

ix = torch.randint(0, Xtr.shape[0], (4,)) # let's look at a batch of just 4 examples
Xb, Yb = Xtr[ix], Ytr[ix]
logits = model(Xb)
print('logits',logits.shape)
print(Xb)
model.layers[0].out.shape # output of embedding layer
model.layers[1].out.shape # output of Flatten layer
model.layers[2].out.shape # output of Linear Layer
# ((torch.randn(4,2,5,80)) @ torch.randn(80,200) + torch.randn(200)).shape
((torch.randn(4,80)) @ torch.randn(80,200) + torch.randn(200)).shape
# (1 2) (3 4) (5 6) (7 8)
((torch.randn(4,4,20)) @ torch.randn(20,200) + torch.randn(200)).shape
list(range(10))[1::2]

for layer in model.layers:
    print(f'{layer.__class__.__name__} : {tuple(layer.out.shape)}')

e = torch.randn(4,8,10) # goal: want this to be (4, 4, 20) where consecutive 10-d vectors get concatenated
explicit = torch.cat([e[:,::2, :], e[:,1::2, :]], dim = 2)
print(explicit.shape)
print((e.view(4,4,20) == explicit).all())

# The same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y
    # logits are the output from the model
    logits = model(Xb)
    # forward pass
    # emb = C[Xb] # embed the characters into vectors
    # x = emb.view(emb.shape[0], -1) # concatenate the vectors
    # x = Xb
    # for layer in layers:
    #     x = layer(x)
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass

    for p in parameters:
        p.grad = None
    loss.backward()

    # update: Simple SGD(Stochastic Gradient Descent)

    lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad


    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    break
plt.plot(lossi)

# put layers into eval mode (needed for batchnorm especially)
for layer in model.layers:
    layer.training = False


# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch

def split_loss(split):
    x,y = {
        'train':(Xtr, Ytr),
        'val':(Xdev, Ydev),
        'test':(Xte, Yte),
    }[split]
    # emb = C[x] # (N, block_size, n_embd)
    # x = emb.view(emb.shape[0], -1) # concat into (N, blocksize * n_embd)    
    # for layer in layers:
    #     x = layer(x)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

# Performance log
# * original (3 character context + 200 hidden neurons, 12k params): train 2,058, val 2, 105
# * context: 3 -> 8(22K params): train 1.918, val 2.027

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        # Forward pass the neural net
        # emb = C[torch.tensor([context])] # (1, block_size, n_embd)
        # x = emb.view(emb.shape[0], -1) # concatenate the vectors
        # for layer in layers:
        #     x = layer(x)
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim = 1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples = 1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break
    
    print(''.join([itos[w] for w in out])) # decode and print the generated word