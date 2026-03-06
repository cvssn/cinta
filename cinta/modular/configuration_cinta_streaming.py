"""configuração do modelo cinta streaming"""

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .configuration_cinta import CintaAcousticTokenizerConfig, CintaDiffusionHeadConfig, _convert_dtype_to_string


logger = logging.get_logger(__name__)

class CintaStreamingConfig(PretrainedConfig):
    model_type = "cinta_streaming"
    
    is_composition = True
    
    sub_configs = {
        "acoustic_tokenizer_config": CintaAcousticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": CintaDiffusionHeadConfig
    }
    
    # keys_to_ignore_at_inference = ["past_key_values"]
    
    # plano tensorial paralelo padrão para o modelo base `qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise"
    }
    
    def __init__(
        self,
        
        acoustic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        tts_backbone_num_hidden_layers=20,
        
        **kwargs
    ):
        # kwargs["_attn_implementation"] = "flash_attention_2"
        kwargs["_attn_implementation_autoset"] = False
        
        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "cinta_acoustic_tokenizer"
            
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, CintaAcousticTokenizerConfig):
            # se uma instância da classe de configuração for fornecida
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            # se um dicionário for fornecido, instanciar a classe de configuração com ele
            # self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
            
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(f"tipo de modelo de decodificador não suportado: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, (Qwen2Config,)):
            # se uma instância da classe de configuração for fornecida
            self.decoder_config = decoder_config

        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "cinta_diffusion_head"
            
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, CintaDiffusionHeadConfig):
            # se uma instância da classe de configuração for fornecida
            self.diffusion_head_config = diffusion_head_config

        # outros parâmetros
        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        
        # o decodificador do modelo é dividido em dois componentes.
        # as camadas inferiores do Transformer são usadas apenas para codificar texto,
        # enquanto as camadas superiores do Transformer são usadas para codificar texto e gerar fala
        # 
        # `tts_backbone_num_hidden_layers` indica o número de camadas superiores utilizadas para tts
        self.tts_backbone_num_hidden_layers = tts_backbone_num_hidden_layers

        super().__init__(**kwargs)
        
    def get_text_config(self, decoder=False):
        """retorna a configuração do decodificador (necessária para compatibilidade de cache com transformers >= 4.57)"""
        
        return self.decoder_config

    @property
    def num_hidden_layers(self):
        """proxy para decoder_config.num_hidden_layers (necessária para transformers >= 4.57)"""
        
        return self.decoder_config.num_hidden_layers

    def to_dict(self):
        """
        substsitui o método to_dict para lidar com a serialização de torch.dtype
        
        corrige: https://github.com/microsoft/VibeVoice/issues/199
        """
        
        output = super().to_dict()
        
        return _convert_dtype_to_string(output)

__all__ = [
    "CintaStreamingConfig"
]