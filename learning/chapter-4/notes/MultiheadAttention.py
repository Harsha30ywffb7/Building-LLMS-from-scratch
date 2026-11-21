import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout,num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads ==0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # need a mask
        self.head_dim = d_out // num_heads
        self.out_proj = nn.Linear(d_out, d_out)
        #initialise weights
        self.wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias = qkv_bias)

        #mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, inputs):
        # inputs now given in batches
        b, num_tokens, d_in = inputs.shape # gets a 3D array
        keys = self.wk(inputs)
        values = self.wq(inputs)
        queries = self.wv(inputs)

        # reshaping the matrix from (b, num_tokens, d_out) to (b, num_tokens, head_num, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposes from shape (b, num_tokens,num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # now find relationships between qkv as previous.
        # Computes dot product for each head
        attention_scores = queries @ keys.transpose(2, 3)
        # mask
        mask_bool = self.mask.bool() [:num_tokens, : num_tokens]

        # apply -ve infinites fill in the upper triangle and add those.
        attention_scores.masked_fill_(mask_bool, -torch.inf)  # mask layer, filling num

        # needed softmax and normalisation
        attention_weights = torch.softmax(attention_scores/keys.shape[-1] ** 0.5, dim=-1)

        # add dropouts
        attention_weights = self.dropout(attention_weights)

        # Tensor shape: (b, num_tokens,n_heads,head_dim)
        context_vec = (attention_weights @ values).transpose(1, 2)

        # adds an optional linear projectiom
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # context vec
        context_vec = self.out_proj(context_vec)

        return context_vec
        