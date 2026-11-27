import torch.nn as nn
import torch
import tiktoken

from MultiheadAttention import MultiHeadAttention
from torch.utils.data import Dataset, DataLoader



# dataloader needs above dataset

def create_dataloader_v1(txt, max_length=256, stride=128, batch_size=4,shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, stride, max_length)
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,     #drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
    num_workers=num_workers # no of cpu processes to use for preprocessing
    )

    return dataloader


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, stride, max_length):
        self.input_ids=[]
        self.target_ids=[]

        token_ids = tokenizer.encode(text)
        # from these token ids create input ids and thier targets.

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk  = token_ids[i: i+max_length] # at a time taking maxlength of inputs and thier target not a single
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids) # returns all rows in the dataset

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx] #Returns a single row from the dataset
    
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
       # we layers
        #token emb, position emb, dropout

        # ok those are embeddings when create embeddings use the Embedding, for inputs use the torch.tensor
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # transformer
        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        ) # here it creates a multiple blocks of the transformers in layers
        #final norm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        #output layer
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # based on index that parts embeddings

        # The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on. 
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device = in_idx.device)
        )
        # in pos_emb an array of sequntial nums are used for the position vector.

        # input level and embeddings
        x = tok_embeds + pos_embeds

        #drop
        x = self.drop_emb(x)

        # transformer block
        x = self.trf_block(x)
        # this x passes through the created transformer blocks in the each layers.

        x = self.final_norm(x) # final layer normalisation to avoid the major differences in the numbers.
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #multi attention
        self.attention = MultiHeadAttention(
             d_in=cfg["emb_dim"],
             d_out=cfg["emb_dim"], 
             context_length=cfg["context_length"], 
             dropout=cfg["drop_rate"],
             num_heads=cfg["n_heads"],
             qkv_bias=cfg["qkv_bias"]
        )
        
        # layernorms
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        #feedforward

        self.ff = FeedForward(cfg)

        #shortcut
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # above transformer diagram in most crucial
        # for first part in the transformer
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x+ shortcut

        # second round here feedforward added
        shortcut =x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4* cfg["emb_dim"],cfg["emb_dim"]),
        )
        #input to output 
        # gelu
        #output to input
    def forward(self, x):
        return self.layers(x)
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True,unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var + self.eps)
        return self.scale * x + self.shift