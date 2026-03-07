import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import AutoModel
from transformers.modeling_utils import PreTrainedModel
# from transformers.modeling_layers import GradientCheckpointLayer
from transformers.activations import ACT2FN
from transformers.utils import logging

from .configuration_cinta import CintaDiffusionHeadConfig


logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
            
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        
        if self.weight is not None:
            output = output * self.weight
            
        return output
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    

def modulate(x, shift, scale):
    """aplica modulação para o tensor de input."""
    
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    incorpora intervalos de tempo escalares em representações vetoriais.
    
    args:
        hidden_size (`int`): tamanho da incorporação de saída
        frequency_embedding_size (`int`, opcional): tamanho da incorporação de frequência intermediária
    """
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            # nn.SiLU(),
            ACT2FN['silu'],
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        
        self.frequency_embedding_size = frequency_embedding_size
        
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        cria incorporações de passos de tempo sinusoidais.
        
        args:
            t (`torch.tensor`): um tensor unidimensional de n índices, um por elemento do lote.
                                esses valores podem ser fracionários.
            dim (`int`): a dimensão da output.
            max_period (`int`, opcional): controla a frequência mínima das inserções.
        
        retorna:
            `torch.tensor`: um tensor [n, d] de embeddings posicionais
        """
        
        half = dim // 2
        
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        
        args = t[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding.to(t.dtype)
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        
        return t_emb
    
    
class FeedForwardNetwork(nn.Module):
    """
    rede feedforward padrão com ativação swiglu.
    
    args:
        embed_dim (`int`): dimensão de entrada
        ffn_dim (`int`): dimensão escondida
    """
    
    def __init__(
        self,
        
        embed_dim,
        ffn_dim
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.act_fn = ACT2FN['silu'] # utilizando silu como função de ativação

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # ativação swiglu
        # gate = F.silu(gate)
        gate = self.act_fn(gate)
        
        return self.down_proj(gate * up)
    

class HeadLayer(nn.Module):
    """
    uma camada na cabeça de difusão.
    
    args:
        embed_dim (`int`): dimensão de entrada
        ffn_dim (`int`): dimensão escondida
        cond_dim (`int`): dimensão de embedding condicional
        norm_eps (`float`, opcional): épsilon para normalização
    """
    
    def __init__(
        self,
        
        embed_dim,
        ffn_dim,
        cond_dim,
        norm_eps=1e-5
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.ffn_dim = ffn_dim
        
        self.ffn = FeedForwardNetwork(self.embed_dim, self.ffn_dim)
        self.norm = RMSNorm(self.embed_dim, eps=norm_eps)
        
        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),
            ACT2FN['silu'],
            nn.Linear(cond_dim, 3 * self.embed_dim, bias=False)
        )

    def forward(self, x, c):
        shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(3, dim=-1)
        
        x = x + gate_ffn * self.ffn(modulate(self.norm(x), shift_ffn, scale_ffn))
        
        return x
    
    
class FinalLayer(nn.Module):
    """
    
    """