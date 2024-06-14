import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32 # how many independent sequences will be process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 200000
eval_interval = 300
# The self attention doesn't tolerate a high learning rate
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # Dim number
# -----------------------------------------------------

torch.manual_seed(1337)

# read it to inspect it
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print("length of dataset in characters: ", len(text))

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
    x, y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()    
    for split in ['train', 'val']:        
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()        
        self.query = nn.Linear(n_embd, head_size, bias = False)        
        self.key = nn.Linear(n_embd, head_size, bias = False)        
        self.value = nn.Linear(n_embd, head_size, bias = False)        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
                

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # Compute attention scores ("affinities")       
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, T, C) --> (B, T, T)        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1)  # (B, T, T)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)        
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)        
        return out


class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-atention in parallel'''

    def __init__(self, num_heads, head_size):
        super().__init__()        
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embd, n_embd)


    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)                
        out = self.proj(out)        
        return out


class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity'''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd))
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''

    def __init__(self, n_embd, n_head):        
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head        
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedForward(n_embd)
        
    def forward(self,x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
    

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_embd//4),
            Block(n_embd, n_embd//4),
            Block(n_embd, n_embd//4),)        
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
       
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        poss_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + poss_emb # (B, T, C)
        x = self.blocks(x)         
        logits = self.ln_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)       

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens                      
            idx_cond = idx[:,-block_size:]            
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)               
            
            probs = F.softmax(logits, dim = -1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx

model = BigramLanguageModel()
# Passing the weigths to cpu o cuda
model = model.to(device = device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype= torch.long, device = device)
print(decode(model.generate(context, max_new_tokens = 100)[0].tolist()))

