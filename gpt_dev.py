# read it to inspect it
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print("length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
print(text[:1000])


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# creating a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda w: [stoi[i] for i in w] # encoder: take a string, output a list of integers
decode = lambda n: ''.join([itos[i] for i in n]) # decoder: take a list of integers, output a string

print(encode('hii there'))
print(decode(encode('hii there')))


# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use Pytorch: https://pytorch.org
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this


# let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90 % will be train, rest val
train_data = data[:n]
val_data = data[n:]


block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
print('longitud de y', len(y))
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'when input is {context} the target: {target}')


torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will be process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs')
print(xb.shape)
print(x)
print('targets:')
print(yb.shape)
print(yb)

print('---------------------')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f'when input is {context.tolist()} the target: {target}')

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            print('logits desde forward: ', logits.shape)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)       

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            # print('Shape de IDX',idx.shape)
            # print('Datos de IDX',idx)
            logits, loss = self(idx)
            # print('------' * 100)
            # print('Logits shape antes en generate',logits.shape)
            # print('Logits',logits)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # print('logits: ', logits.shape)
            # print('------' * 100)
            # print('Logits shape despues en generate',logits.shape)
            # print('Logits',logits)
            # print('------' * 100)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (1,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx



m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(torch.zeros((1,1), dtype = torch.long), max_new_tokens = 10)[0].tolist()))

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
# for steps in range(20000):
#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none = True)
#     loss.backward()
#     optimizer.step()

# print(loss.item())

# print(decode(m.generate(torch.zeros((1,1), dtype = torch.long), max_new_tokens = 100)[0].tolist()))

# The mathematical trick in self-attention

# consider the following toy example:
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)
print('X:', x)
print('Dimension X:',x.shape)
print('Una fila de X', x[0, :1 + 1])
print('Dimension de una fila de X', x[0,0].shape)


# We want x[b,t] = mean{i<=t} x{b,i}
# This is a inefficient way
# Versión 1
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C)
        # print('Xprev',xprev)
        xbow[b,t] = torch.mean(xprev, 0)
        # print('Xbow', xbow)

# The trick is this:
# Versión 2
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a,1, keepdim = True)
# a = torch.ones(3,3)
b = torch.randint(0,10, (3,2)).float()
c = a @ b
print('a=')
print(a)
print('----')
print('b=')
print(b)
print('----')
print('c=')
print(c)

wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim = True)
print('Triangulo inferior',wei)
xbow2 = wei @ x # (B, T, T) @ (B, T,C) ----> (B, T, C) # Python crea B en wei para igualar dim
print('Xbow and Xbow3 are the same or aproximate',torch.allclose(xbow, xbow2))

# Versión 3: use Softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = 1)
xbow3 = wei @ x
print('Xbow and Xbow3 are the same or aproximate',torch.allclose(xbow, xbow3))

# Versión 4: Self-attention!
torch.manual_seed(1337)
B, T, C = 4, 8, 32

# Let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias = False)
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k = key(x)     # (B, T, C)
q = query(x)   # (B, T, C)
v = value(x)   # (B, T, C)
wei = q @ k.transpose(-2,-1)  # (B, T, 16) @ (B, 16, T) ----> (B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = 1)
out = wei @ v

# Notes:
# * Attention is a communication mechanism. Can be seen as nodes in a directed graph
# looking at each other and aggregating information with a weighted sum from all nodes
# that point to them, with data-dependent weights.
# * There is no notion of space. Attention simply acts over a set of vectors. This is why
# we need to positionally encode tokens.
# * Each example across batch dimension is of course processed completely independently
# and never 'talk' to each other
# In an 'encoder' attention block just delete the line does masking with tril, allowing
# all tokens to communicate. This block here is called 'decoder' attention block because
# it has triangular masking, and is usually used in autoregressive settings, like 
# language modeling
# "Self-attention" just means that the keys and values are produced from the same source
# as queries. In 'Cross-attention', the queries still get produced from x, but the keys
# and values come from some other, external source (e.g and encoder module)
# "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when
# input Q, K are unit variance, wei will be unit variance too and Softmax will stay diffuse
# and not saturate too much. Illustration below

k = torch.randn(B,T, head_size)
q = torch.randn(B,T, head_size)
wei = q @ k.transpose(-2, -1) * head_size **-5

print('Varianza key',k.var())
print('Varianza query',q.var())
print('Varianza wey', wei.var())

hot_soft = torch.tensor([0.1,-0.2,0.3,-0.2,0.5])
print('Shape of hot_soft', hot_soft.shape)
print(torch.softmax(hot_soft, dim = -1))
print(torch.softmax(hot_soft, dim = -1)*8)

