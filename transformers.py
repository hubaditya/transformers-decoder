import torch
import torch.nn as nn
from torch.nn import functional as F


BATCH_SIZE = 32 # how many examples to process at once
BLOCK_SIZE = 8 # how many previous characters to take to predict the next character
MAX_ITERS = 5000 # total number of iterations during training
EMBEDDING_SIZE = 32 # size of word and positional embeddings
HEAD_SIZE = 32 # head size for self-attention
NUM_HEAD = 4 # number of heads for multi-attention
NUM_BLOCKS = 6 # number of blocks to replicate
EVAL_INTERVAL = 500 # after how many iterations should the result be printed
EVAL_SIZE = 200 # how many examples should be taken to compute training and validation loss
learning_rate = 1e-3
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)


with open("input.txt", "r") as file:
    data = file.read()


def get_chars(data):
    return sorted(list(set(data)))


def get_vocab_size(chars):
    return len(chars) 


def encode(encoding_dict, string):
    return [encoding_dict[char] for char in string]


def decode(decoding_dict, idxs):
    return "".join([decoding_dict[idx] for idx in idxs])


def train_val_split(data, split_ratio=0.9):
    sample_size = int(len(data) * split_ratio)
    train_data = data[:sample_size]
    val_data = data[sample_size:]
    return train_data, val_data


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad() # letting pytorch know no backward computation needs to happen
def estimate_loss(model, train_data, val_data):
    out = {}
    # setting this to validation mode. This becomes necessary when for example there are dropout layers. During validation, all parameters need to be used
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_SIZE)
        data = train_data if split == "train" else val_data
        for i in range(EVAL_SIZE):
            xb, yb = get_batch(data)
            _, loss = model.forward(xb, yb)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


chars = get_chars(data)
vocab_size = get_vocab_size(chars)
encoding = {c:i for i, c in enumerate(chars)}
decoding = {i:c for i, c in enumerate(chars)}
tensor_data = torch.tensor(encode(encoding, data), dtype=torch.long)
train_data, val_data = train_val_split(tensor_data)


class Head(nn.Module):
    
    def __init__(self, HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.query = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        # since "tril" is not a model parameter, pytorch by default will put it on CPU
        # by registering this as a buffer, pytorch puts this on GPU 
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, E = x.shape
        k = self.key(x) # shape = (B, T, H)
        q = self.query(x) # shape = (B, T, H)
        v = self.query(x) # shape = (B, T, H)
        wgt = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, H) and (B, H, T) give shape = (B, T, T) 
        # masking the future tokens not to interact with the current token. Only the past should interact
        # during training, [:T, :T] won't be useful because we will have BLOCK_SIZE data. 
        # But during text generation we will be starting with [1, 1]
        wgt = wgt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wgt = F.softmax(wgt, dim=-1)
        wgt = self.dropout(wgt)
        out = wgt @ v # (B, T, T) and (B, T, H) give shape = (B, T, H)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, NUM_HEAD, HEAD_SIZE):
        super().__init__()
        head_subsize = HEAD_SIZE // NUM_HEAD
        self.heads = nn.ModuleList([Head(head_subsize) for _ in range(NUM_HEAD)])
        # used only to perform addition for residual connection
        self.proj = nn.Linear(HEAD_SIZE, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # shape of x = (B, T, E)
        out = torch.cat([h.forward(x) for h in self.heads], dim=-1) # shape = (B, T, H)
        out = self.proj(out) # shape = (B, T, E)
        return self.dropout(out)
    

class FeedForward(nn.Module):

    def __init__(self, EMBEDDING_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE),
            nn.ReLU(),
            # used only to perform addition for residual connection
            nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # shape of x = (B, T, E) 
        return self.net(x) # shape = (B, T, E)
    

class Block(nn.Module):

    def __init__(self, NUM_HEAD, HEAD_SIZE, EMBEDDING_SIZE):
        super().__init__()
        # creating 4 attention modules with 8 dimension vector each to give a total of 32
        self.ln1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.sa_head = MultiHeadAttention(NUM_HEAD, HEAD_SIZE)
        self.ln2 = nn.LayerNorm(EMBEDDING_SIZE)
        self.ffwd = FeedForward(EMBEDDING_SIZE)

    def forward(self, x):
        # shape of x = (B, T, E)
        # adding x to the attention and feed-forward blocks as a residual connection
        x = self.ln1(x)
        x = x + self.sa_head.forward(x) # shape = (B, T, E)
        x = self.ln2(x)
        x = x + self.ffwd.forward(x) # shape = (B, T, E)
        return x


class BiGramLangModel(nn.Module):

    def __init__(self):
        super().__init__()
        # initializes an embedding with (C, E) shape
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        # initializes an embedding with (T, E) shape
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_SIZE)
        self.block = nn.Sequential(*[Block(NUM_HEAD, HEAD_SIZE, EMBEDDING_SIZE) for _ in range(NUM_BLOCKS)])
        self.ln_f = nn.LayerNorm(EMBEDDING_SIZE)
        self.lm_head = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets shape = (B, T) where B = BATCH_SIZE and T = BLOCK_SIZE
        B, T = idx.shape
        # passing indexes to the embedding table to get their respective embeddings
        token_embed = self.token_embedding_table(idx) # input = (B, T) and output = (B, T, E)
        # we initialize a tensor everytime because we only care about the position
        position_embed = self.position_embedding_table(torch.arange(T, device=device)) # input = (T) and output = (T, E)
        x = token_embed + position_embed # (B, T, E) + (T, E) = (B, T, E) by automatic broadcasting
        x = self.block.forward(x) # shape = (B, T, E)
        x = self.ln_f(x) # shape = (B, T, E)
        logits = self.lm_head(x) # shape = (B, T, C) where C = VOCAB_SIZE
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # changing shape to compute loss
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_output_token):
        # idx shape = (B, T)
        for _ in range(max_output_token):
            # when calling position embedding in the forward function it takes BLOCK_SIZE as input
            idx_subset = idx[:, -BLOCK_SIZE:] # passing only last BLOCK_SIZE values
            logits, _ = self.forward(idx_subset) # shape = (B, T, C)
            # taking the last element (the last generated char) along time axis
            logits = logits[:, -1, :] # shape = (B, C)
            # find softmax along C
            probs = F.softmax(logits, dim=-1) # shape = (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # shape = (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # shape = (B, T+1) for every iteration
        return idx # shape = (B, T+max_output_token)


def train(model, params, train_data, val_data, learning_rate):
    optimizer = torch.optim.AdamW(params, learning_rate)
    for steps in range(MAX_ITERS):
        if steps % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step: {steps} -> Training Loss: {losses['train']:.4f} "
                  + f"and Validation Loss: {losses['val']:.4f}")
        xb, yb = get_batch(train_data)
        _, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True) # ensuring no gradient accumulation
        loss.backward()
        optimizer.step()


model = BiGramLangModel()
m = model.to(device)
train(m, m.parameters(), train_data, val_data, learning_rate)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(decoding, m.generate(context, max_output_token=500)[0].tolist()))
