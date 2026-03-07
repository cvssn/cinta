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
        embed_dim (`int`): dimensão de entrada.
        ffn_dim (`int`): dimensão escondida.
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
        # gate = f.silu(gate)
        gate = self.act_fn(gate)
        
        return self.down_proj(gate * up)
    

class HeadLayer(nn.Module):
    """
    uma camada na cabeça de difusão.
    
    args:
        embed_dim (`int`): dimensão de entrada.
        ffn_dim (`int`): dimensão escondida.
        cond_dim (`int`): dimensão de embedding condicional.
        norm_eps (`float`, opcional): épsilon para normalização.
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
    camada final na cabeça de difusão.
    
    args:
        hidden_size (`int`): dimensão de entrada.
        output_size (`int`): dimensão de saída.
        cond_size (`int`): dimensão de incorporação da condição.
        norm_eps (`float`, opcional): épsilon para normalização.
    """
    
    def __init__(self, hidden_size, output_size, cond_size, norm_eps=1e-5):
        super().__init__()
        
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        
        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),
            ACT2FN['silu'],
            nn.Linear(cond_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        
        return x
    
    
class CintaDiffusionHead(PreTrainedModel):
    """
    modelo de cabeçote de difusão para cinta.
    
    args:
        config (`CintaDiffusionHeadConfig`): configuração de modelo.
        latent_size (`int`, opcional): tamanho do espaço latente. se não for fornecido, utiliza `config.latent_size`.
    """
    
    config_class = CintaDiffusionHeadConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    
    def __init__(
        self,
        
        config
    ):
        super().__init__(config)
        
        self.config = config
        self.cond_dim = config.hidden_size
        latent_size = config.latent_size
        
        self.noisy_images_proj = nn.Linear(latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(config.hidden_size, self.cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.cond_dim)
        
        ffn_dim = int(config.hidden_size * config.head_ffn_ratio)
        
        # cria as camadas intermediárias
        self.layers = nn.ModuleList([
            HeadLayer(
                embed_dim=config.hidden_size,
                ffn_dim=ffn_dim,
                cond_dim=self.cond_dim,
                norm_eps=config.rms_norm_eps
            )
            
            for _ in range(config.head_layers)
        ])
        
        # camada final para saída
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size, 
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=config.rms_norm_eps
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        """inicializa os pesos do modelo."""
        
        # inicializa o incorporador de passo de tempo
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # camadas de modulação adaln zeradas
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)

        # camadas de saída zeradas
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(
        self,
        
        noisy_images,
        timesteps,
        condition
    ):
        """
        passe para frente da cabeça de previsão.
        
        args:
            noisy_images (`torch.tensor`): imagens ruidosas/latentes para remover ruído.
            timesteps (`torch.tensor`): etapas de tempo para difusão.
            condition (`torch.tensor`): informações de condicionamento.
            
        returna:
            `torch.tensor`: o ruído/velocidade previsto.
        """
        
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        
        condition = self.cond_proj(condition)
        c = condition + t
        
        for layer in self.layers:
            x = layer(x, c)
            
        x = self.final_layer(x, c)
        
        return x


AutoModel.register(CintaDiffusionHeadConfig, CintaDiffusionHead)

__all__ = [
    "CintaDiffusionHead"
]