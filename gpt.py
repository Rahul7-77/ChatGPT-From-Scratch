import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
# max_iters = 3000
epochs = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 384
n_layer = 6
n_head = 6
dropout = 0.2
# ------------

# for reproducing same values
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
# Encode -> Takes a string and outputs a list of integers
encode = lambda s: [stoi[ch] for ch in s]
# Decode -> Takes a list of integers and outputs string
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0,len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x,y = x.to(device),y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # Setting model to evaluation mode to make sure all layers are in inference mode. for example for batchnorm or layernorm
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Setting model back to training mode to enable backprop and gradient descent
    model.train()
    return out

class Head(nn.Module):
    """ one head of self attention """
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_emb,head_size,bias=False)
        self.query = nn.Linear(n_emb,head_size,bias=False)
        self.value = nn.Linear(n_emb,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,context):
        B,T,C = context.shape # T is block_size and C is n_emb
        k = self.key(context) #(B,T,head_size)
        q = self.query(context) #(B,T,head_size)
        v = self.value(context) #(B,T,head_size)
        # tril = torch.tril(torch.ones(T,T))
        wei = q @ k.transpose(1,2) * k.shape[-1]**-0.5 # (B,T,head_size) @ (B,head_size,T) --> (B,T,T). For now we are taking head size as n_emb so we are doing *C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1) # (B,T,T) normalised
        wei = self.dropout(wei)
        out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self attention in parallel """
    def __init__(self,head_size,num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Adding a projection layer which is nothing but transforming the output into linear fashion
        self.proj = nn.Linear(head_size*num_heads,n_emb)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # Just concat all the results of heads in MultiHeadAttention block
        out = torch.cat([h(x) for h in self.heads],dim=-1) # we are concatinating over last dimension which is channels dimension or n_emb dimension
        # why concat is happening in above code across channel dimension? Because we are taking head_size as n_emb/num_heads for every individual head block and we are concatinating everything to again get C
        out = self.proj(out) # projecting into linear fashion
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ Just a simple neural layer with a non linearity """
    def __init__(self,n_emb):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_emb,4*n_emb),
            nn.ReLU(),
            # adding projection layer
            nn.Linear(4*n_emb,n_emb),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.ff(x)

class Block(nn.Module):
    """ Transformer block : Multiple multiheadattention blocks along with a feedforward neural layer"""
    def __init__(self,n_emb,num_heads):
        super().__init__()
        head_size = n_emb//num_heads
        self.mha = MultiHeadAttention(head_size,num_heads)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
    def forward(self,x):
        # Here we are adding skip connections or residual connections. The blocks are like some other branch that got deviated from main branch
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GptModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create embedding or lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_emb)
        self.position_embedding_table = nn.Embedding(block_size,n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb,num_heads=n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb,vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,inputs,targets=None):
        # inputs and targets both are (B,T) tensors. Here B is batch size and T is block_size or block_length
        B,T = inputs.shape
        tok_emb = self.token_embedding_table(inputs) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # This will be (T,C)
        x = tok_emb+pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x) # (B,T,Vocab_size)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            # Now logits is of shape (B,T,C). Targets are already in shape of (B,T). now to calculate loss we need (B*T,C) for logits and (B*T) for targets
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self,context,max_new_tokens):
        # Context is of shape (B,T)
        for _ in range(max_new_tokens):
            # since we are using positional encoding and block size, we will be taking last block size tokens
            context_crop = context[:,-block_size:]
            logits,loss = self(context_crop)
            # logits is shape (B,T,C). We want to get last token to generate. so we need last token in all batches
            logits = logits[:,-1,:] # logits shape will be (B,C) now
            probs = F.softmax(logits,dim=-1)
            next_idx = torch.multinomial(probs,num_samples=1) # This will be a (B,1) tensor
            # Excellent thing here we can see batches are independent and we can see parallel computation
            context = torch.cat((context,next_idx),dim=1)
        return context

model = GptModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Creating a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    # for every 500 epochs estimate loss
    if epoch % eval_interval == 0 or epoch == epochs-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # Get batch data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generating text from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))