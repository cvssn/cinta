#!/usr/bin/env python

# coding=utf-8

import argparse
import json
import os
from pathlib import Path
import re
import torch
from typing import Dict, List, Tuple

from cinta.modular.configuration_cinta import (CintaConfig)
from cinta.modular.modeling_cinta import CintaForConditionalGeneration
from transformers.utils import logging


logger = logging.get_logger(__name__)

def convert_cinta_nnscaler_checkpoint_to_hf(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    
    config_path: str = None
):
    """
    converte um checkpoint cinta do nnscaler para o formato huggingface.
    suporta checkpoints regulares e checkpoints paralelos de tensores
    """
    
    # carrega um checkpoint regular
    logger.info(f"carregando checkpoint regular de {checkpoint_path}")
    
    # # ['model', 'optimizer', 'lr_scheduler', 'train_status', 'train_args', 'rng_states', 'nnscaler', 'dataloader']
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # config = checkpoint['train_args']
    init_config_name = checkpoint['train_args']['vars']['model_args']['config_path']['relative_path']
    pretrained_name = checkpoint['train_args']['vars']['data_args']['tokenizer_path']
    
    init_config_path = Path(__file__).parent.parent / 'configs' / init_config_name.split('/')[-1]
    
    if init_config_path.exists():
        logger.info(f"carregando a configuração inicial de {init_config_path}")
        
        with open(init_config_path, 'r') as f:
            init_config = json.load(f)
    else:
        raise FileNotFoundError(f"arquivo de configuração inicial {init_config_path} não encontrado. forneça um path válido")

    tie_word_embeddings = init_config['decoder_config'].get('tie_word_embeddings', True)
    
    logger.info(f"embeddings de palavras tie: {tie_word_embeddings}")
    
    init_config['decoder_config']['use_cache'] = True
    
    config = CintaConfig(**init_config, tie_word_embeddings=tie_word_embeddings)
    
    # extrai
    model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in checkpoint["model"].items() if k.startswith('model.model.')}
    
    if not tie_word_embeddings and 'model.lm_head.weight' in checkpoint["model"].keys():
        # caso não haja vinculação de pesos, precisamos adicionar o peso lm_head separadamente
        model_state_dict['lm_head.weight'] = checkpoint["model"]['model.lm_head.weight']
        
    # substitui pela configuração fornecida, se disponível
    if config_path:
        logger.info(f"carregando configuração de {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = CintaConfig.from_dict(config_dict)
        
    # define o tipo de dados padrão como bfloat16 antes de criar o modelo
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    
    # cria o modelo huggingface
    logger.info("criando modelo cintaforconditionalgeneration do huggingface")
    model = CintaForConditionalGeneration(config)
    
    # restaura o dtype original
    torch.set_default_dtype(original_dtype)
    
    # carrega o dicionário de estado
    logger.info("Loading weights into model")
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
        
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
        
    # cria o diretório de output
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    
    # salva o modelo e a configuração
    logger.info(f"Saving model to {pytorch_dump_folder_path}")
    
    # salva a configuração
    config.save_pretrained(pytorch_dump_folder_path)
    
    # salva a configuração cintaprocessor
    logger.info("salvando configuração do cintaprocessor")
    
    processor_config = {
        "processor_class": "CintaProcessor",
        "speech_tok_compress_ratio": 3200,
        "db_normalize": True,
        
        # configuração do processador de áudio
        "audio_processor": {
            "feature_extractor_type": "CintaTokenizerProcessor",
            "sampling_rate": 24000,
            "normalize_audio": True,
            "target_dB_FS": -25,
            "eps": 1e-6
        },
        
        "language_model_pretrained_name": pretrained_name
    }
    
    processor_config_path = os.path.join(pytorch_dump_folder_path, "preprocessor_config.json")
    
    with open(processor_config_path, 'w') as f:
        json.dump(processor_config, f, indent=2)
    
    logger.info(f"configuração do processador salva em {processor_config_path}")
    
    # salva o modelo com sharding
    #
    # save_pretrained manuseia os pesos amarrados automaticamente
    logger.info("salvando pesos de modelos com fragmentação...")
    
    model.save_pretrained(
        pytorch_dump_folder_path,
        max_shard_size="2GB", # Set maximum size for each shard
        safe_serialization=True # Ensure saving in .safetensors format
    )
    
    logger.info(f"pesos do modelo salvos em {pytorch_dump_folder_path}")
    
    logger.info("conversão concluída!")
    
    # verifica se o modelo salvo pode ser carregado
    logger.info("verificando modelo salvo...")
    loaded_model = CintaForConditionalGeneration.from_pretrained(pytorch_dump_folder_path)
    logger.info("modelo carregado com sucesso a partir do checkpoint salvo!")
    
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--nnscaler_checkpoint_path",
        
        type=str,
        required=True,
        
        help="path para o checkpoint fairseq (arquivo .pt). para checkpoints paralelos de tensores, "
             "forneça qualquer um dos arquivos de peça. (exemplo, checkpoint_1_5000-model_part-0.pt), "
             "e o script irá detectar e mesclar automaticamente todas as partes"
    )
    
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        
        type=str,
        required=True,
        
        help="path para o diretório do modelo pytorch de saída"
    )
    
    parser.add_argument(
        "--config_path",
        
        type=str,
        default=None,
        
        help="path opcional para um arquivo json de configuração para substituir a configuração extraída"
    )
    
    args = parser.parse_args()
    
    convert_cinta_nnscaler_checkpoint_to_hf(
        args.nnscaler_checkpoint_path,
        args.pytorch_dump_folder_path,
        
        args.config_path
    )
    
    
if __name__ == "__main__":
    main()