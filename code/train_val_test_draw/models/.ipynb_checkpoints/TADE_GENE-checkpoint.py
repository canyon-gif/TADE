import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

# Arithmetic Block with Prompt Optimization and Dropout
class ArithmeticBlock(nn.Module):
    def __init__(self, token_num, heads, dim, attn_dropout, k_sum, k_prod, use_prompts=True, qk_relu=False):
        super().__init__()
        self.use_prompts = use_prompts
        self.attn_dropout = attn_dropout
        self.qk_relu = qk_relu

        self.heads = heads
        self.head_dim = dim // heads  

        if self.use_prompts:
            self.sum_prompt = nn.Parameter(torch.randn(1, k_sum, dim))
            self.prod_prompt = nn.Parameter(torch.randn(1, k_prod, dim))
        else:
            self.sum_WQ = nn.Linear(dim, dim, bias=False)
            self.sum_WK = nn.Linear(dim, dim, bias=False)
            self.sum_WV = nn.Linear(dim, dim, bias=False)
            self.prod_WQ = nn.Linear(dim, dim, bias=False)
            self.prod_WK = nn.Linear(dim, dim, bias=False)
            self.prod_WV = nn.Linear(dim, dim, bias=False)

        self.sum_out = nn.Linear(dim, dim)
        self.prod_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        if self.use_prompts:
            sum_q = repeat(self.sum_prompt, '1 k d -> b k d', b=b)
            sum_k = x
            sum_v = x
        else:
            sum_q = self.sum_WQ(x)
            sum_k = self.sum_WK(x)
            sum_v = self.sum_WV(x)
        sum_q = rearrange(sum_q, 'b n (h d) -> b h n d', h=self.heads)
        sum_k = rearrange(sum_k, 'b n (h d) -> b h n d', h=self.heads)
        sum_v = rearrange(sum_v, 'b n (h d) -> b h n d', h=self.heads)

        if self.qk_relu:
            sum_attn = F.softmax((sum_q * sum_k).sum(-1) / self.head_dim**0.5, dim=-1)
        else:
            sum_attn = F.softmax((sum_q @ sum_k.transpose(-1, -2)) / self.head_dim**0.5, dim=-1)
        sum_attn = F.dropout(sum_attn, p=self.attn_dropout, training=self.training)

        sum_out = sum_attn @ sum_v
        sum_out = rearrange(sum_out, 'b h n d -> b n (h d)')   
        sum_out = self.sum_out(sum_out)

        if self.use_prompts:
            prod_q = repeat(self.prod_prompt, '1 k d -> b k d', b=b)
            prod_k = torch.log(F.relu(x) + 1e-8)
            prod_v = torch.log(F.relu(x) + 1e-8)
        else:
            prod_q = self.prod_WQ(torch.log(F.relu(x) + 1e-8))
            prod_k = self.prod_WK(torch.log(F.relu(x) + 1e-8))
            prod_v = self.prod_WV(torch.log(F.relu(x) + 1e-8))
        
        prod_q = rearrange(prod_q, 'b n (h d) -> b h n d', h=self.heads)
        prod_k = rearrange(prod_k, 'b n (h d) -> b h n d', h=self.heads)
        prod_v = rearrange(prod_v, 'b n (h d) -> b h n d', h=self.heads)

        if self.qk_relu:
            prod_attn = F.softmax((prod_q * prod_k).sum(-1) / self.head_dim**0.5, dim=-1)
        else:
            prod_attn = F.softmax((prod_q @ prod_k.transpose(-1, -2)) / self.head_dim**0.5, dim=-1)
        prod_attn = F.dropout(prod_attn, p=self.attn_dropout, training=self.training)

        prod_out = prod_attn @ prod_v
        prod_out = rearrange(prod_out, 'b h n d -> b n (h d)')   
        prod_out = self.prod_out(torch.exp(prod_out))

        out = torch.cat([sum_out, prod_out], dim=1)
        return out
    

class GenePredictor(nn.Module):
    def __init__(self, gene_dim, text_dim, dim, depth, heads, attn_dropout, ff_dropout, k_sum, k_prod, use_prompts=True, qk_relu=False):
        super().__init__()
        self.gene_dim = gene_dim
        self.text_dim = text_dim
        
        self.gene_embedding = nn.Linear(gene_dim, dim)
        self.text_embedding = nn.Linear(text_dim, dim)

        self.gene_layers = nn.ModuleList([
            ArithmeticBlock(token_num=gene_dim, heads=heads, dim=dim,
                            attn_dropout=attn_dropout, k_sum=k_sum,
                            k_prod=k_prod, use_prompts=use_prompts,
                            qk_relu=qk_relu)
            for _ in range(depth)
        ])

        self.text_layers = nn.ModuleList([
            ArithmeticBlock(token_num=text_dim, heads=heads, dim=dim,
                            attn_dropout=attn_dropout, k_sum=k_sum,
                            k_prod=k_prod, use_prompts=use_prompts,
                            qk_relu=qk_relu)
            for _ in range(depth)
        ])


        self.fusion_layer = nn.Linear(2 * dim * (k_sum + k_prod), dim)

        self.output_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):   
        gene_x = x[:, :self.gene_dim]
        text_x = x[:, self.gene_dim:]

        gene_x = self.gene_embedding(gene_x).unsqueeze(1)
        text_x = self.text_embedding(text_x).unsqueeze(1)

        for layer in self.gene_layers:
            gene_x = layer(gene_x)

        for layer in self.text_layers:
            text_x = layer(text_x)

        x = torch.cat([gene_x.view(gene_x.size(0), -1), text_x.view(text_x.size(0), -1)], dim=1)
        x = self.fusion_layer(x)

        logit = self.output_layer(x)

         
        loss = None
        if label is not None:
            loss = nn.BCELoss()(logit.reshape(-1), label.float().reshape(-1))

        return logit if loss is None else (logit, loss)
