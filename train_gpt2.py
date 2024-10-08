from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import transformers
import code
import time


# -------------------------------------------------------------- #
batch_size = 64 # how many independent sequences will we process
block_size = 256 # what is the maximum context length for predict
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

gpt = GPTConfig()
print(gpt)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        print("self.transformer.wte before: ", self.transformer.wte.weight.shape)
        print("self.lm_head: ", self.lm_head.weight.shape)
        
        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        print("self.transformer.wte after: ", self.transformer.wte)
        print(torch.all(self.transformer.wte.weight == self.lm_head.weight))

        # init params
        self.apply(self.__ini__weigths)

    def __ini__weigths(self, module):
        
        if isinstance(module, nn.Linear):            
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std = (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        # idx is of shape (B,T)      
        # print("Entre al modelo", idx, "shape: ", idx.shape)  
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # shape (T)
        # print("llegue al pos", T)
        pos_emb = self.transformer.wpe(pos) # posisition embeddings of shape (T, n_embd)
        # print("Pos_emb", self.transformer.wte)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # print("Tok_emb")
        x = tok_emb + pos_emb
        # print("LLegue al posicion embedding ")
        # forward the blocks of the transformer
        for block in self.transformer.h:
                x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        print("Logits shape: ", logits.shape)        
        loss = None
        if targets is not None:
            print("Targets shape: ", targets.shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # print("LLegue a los logits")
        return logits, loss
    

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weigths from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gtp2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('Loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2' : dict(n_layer = 12, n_head = 12, n_embd = 768), # 124M params
            'gpt2-medium' : dict(n_layer = 24, n_head = 16, n_embd = 1024), # 350M params
            'gpt2-large' : dict(n_layer = 36, n_head = 20, n_embd = 1280), # 774M params
            'gpt2-xl' : dict(n_layer = 48, n_head = 25, n_embd = 1600), # 1558M params
        }[model_type]
        print("forcing vocab_size = 50257, block_size = 1024, bias = True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # print('******* GPT 2 de Hugging Face *******')
        # for k, v in sd_hf.items():
        #     print(k, v.shape)

        print('******* GPT 2 de la PC*******')
        for k, v in sd.items():
            print(k, v.shape)


        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight',  'mlp.c_proj.weight']

        # basically the openai checkpoints use a 'Conv1D' module, but we only want to use a vanilla Linear

        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # print("Entre al if")
                # special treatment for the Conv1D weights we need to transpose
                # print("Key: ", k)
                # print('Transposed: ', 'HF', sd_hf[k].shape, 'PC', sd[k].shape)
                # print("HF shape", sd_hf[k].shape[::-1])
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
                    
            else:
                # print("Entre al else")
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn : p for pn, p in self.named_parameters()}


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


    

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
    
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimentionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be batch
        # nh is 'number of heads', hs is 'head size', and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124 M), n_head = 12, hs = 64, nh * hs = C = 768 channels in the transformer
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim = -1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side 
        # output projection
        y = self.c_proj(y)
        return y


# ----------------------------------------------------------
# Generation
# num_return_sequences = 5
# max_length = 30
# model = GPT.from_pretrained('gpt2')
# print("didn't crash yay!")

# model.eval()
# model.to('cpu')

# # prefix tokens 
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# print('tokens encode: ', tokens)
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# print("Tokens en tensor: ", tokens, 'Shape: ', tokens.shape)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
# print("tokens unqueeze repeat: ", tokens, 'Shape: ',tokens.shape)
# x = tokens.to('cpu')

# # generate: right now x is (B, T) where B = 5, T = 8
# # set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         print("Shape logits: ", logits.shape)
#         # take the logits at the last position
#         logits = logits[:,-1,:] # (B, vocab_size)
#         print("(B, vocab sizes): ", logits.shape)
#         # get the probabilities
#         probs = F.softmax(logits, dim = -1)

#         # do top-k sampling of 50 (hugging pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
#         # select a token from the top-k probabilities
#         print("topk_probs shape: ", topk_probs.shape)
#         ix = torch.multinomial(topk_probs, 1) # (B,1)
#         print("ix shape: ", ix.shape)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         print("xcol: ", xcol)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim = 1)


# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     print("tokens: ", tokens)
#     decoded = enc.decode(tokens)
#     print(">", decoded)

# ----------------------------------------------------------
# Traine
# attempt to autodetect the device
# device = "cpu"
# if torch.cuda.is_available():
#     device ="cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
#     device = "mps"
# print("using device: ", device)




# num_return_sequences = 5
# max_length = 30
# model = GPT(GPTConfig())
# print("didn't crash yay!")

# model.eval()
# model.to(device)

# # prefix tokens 
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# print('tokens encode: ', tokens)
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# print("Tokens en tensor: ", tokens, 'Shape: ', tokens.shape)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
# print("tokens unqueeze repeat: ", tokens, 'Shape: ',tokens.shape)
# x = tokens.to('cpu')

# # generate: right now x is (B, T) where B = 5, T = 8
# # set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         # print("Shape logits: ", logits.shape)
#         # take the logits at the last position
#         logits = logits[:,-1,:] # (B, vocab_size)
#         # print("(B, vocab sizes): ", logits.shape)
#         # get the probabilities
#         probs = F.softmax(logits, dim = -1)

#         # do top-k sampling of 50 (hugging pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
#         # select a token from the top-k probabilities
#         # print("topk_probs shape: ", topk_probs.shape)
#         ix = torch.multinomial(topk_probs, 1) # (B,1)
#         # print("ix shape: ", ix.shape)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # print("xcol: ", xcol)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim = 1)


# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     print("tokens: ", tokens)
#     decoded = enc.decode(tokens)
#     print(">", decoded)

# -----------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory

        with open("input.txt","r") as f: 
            text = f.read()
        
        print("text", text[:100])
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens)")
        print(f"1 epoch = {len(self.tokens) //  (B*T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
# ----------------------------------------------------------
# Traine
# attempt to autodetect the device
# device = "cpu"
if torch.cuda.is_available():
    device ="cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
    device = "mps"
print("using device: ", device)

device = "cpu" # OVERRIDE

num_return_sequences = 5
max_length = 30
model = GPT(GPTConfig(vocab_size=50304))
print("didn't crash yay!")

model.eval()
model.to(device)

# prefix tokens 
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open("input.txt", 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B = 4, T = 8)
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
# logits, loss  = model(x, y)
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4,eps=1e-8, betas=(0.9,0.95))
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device,dtype= torch.bfloat16):
        logits, loss = model(x, y)
        
    # code.interact(local=locals())
    loss.backward()
    # normalize the params and return the norm of the params
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    # torch.cuda.synchronize()
    t1 =  time.time()
    dt = (t1 - t0) * 1000 # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"Step {step} | loss: {loss.item()} | lr {lr:.4e} | norm : {norm:.4f} | dt: {dt:.2f} ms, tok/sec: {tokens_per_sec:.2f}")


# print(logits.shape)
print("Loss: ", loss)
import sys; sys.exit(0)

# ------------------------------------------------------------------------------------------


print('tokens encode: ', tokens)
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
print("Tokens en tensor: ", tokens, 'Shape: ', tokens.shape)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
print("tokens unqueeze repeat: ", tokens, 'Shape: ',tokens.shape)
x = tokens.to('cpu')

# generate: right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # print("Shape logits: ", logits.shape)
        # take the logits at the last position
        logits = logits[:,-1,:] # (B, vocab_size)
        # print("(B, vocab sizes): ", logits.shape)
        # get the probabilities
        probs = F.softmax(logits, dim = -1)

        # do top-k sampling of 50 (hugging pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
        # select a token from the top-k probabilities
        # print("topk_probs shape: ", topk_probs.shape)
        ix = torch.multinomial(topk_probs, 1) # (B,1)
        # print("ix shape: ", ix.shape)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # print("xcol: ", xcol)
        # append to the sequence
        x = torch.cat((x, xcol), dim = 1)


# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    print("tokens: ", tokens)
    decoded = enc.decode(tokens)
    print(">", decoded)