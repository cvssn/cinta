"""mapeador de entrada de áudio para o pipeline multimodal vllm.

este módulo lida com o carregamento e pré-processamento de dados de áudio para inferência de reconhecimento automático de fala (asr) do cinta.

ele converte vários formatos de entrada de áudio (path, bytes, array numpy) em tensores
que podem ser processado pelo modelo cinta
"""

import torch
import numpy as np
from typing import Union, List
from vllm.multimodal.inputs import MultiModalInputs
from cinta.processor.audio_utils import load_audio_use_ffmpeg, load_audio_bytes_use_ffmpeg, AudioNormalizer


def load_audio(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    """carrega e normaliza um áudio a partir de um path de arquivo.
    
    args:
        audio_path: path para o arquivo de áudio
        target_sr: taxa de amostragem desejada para o áudio carregado (padrão: 24khz para o cinta)
    
    retorna:
        forma de onda do áudio normalizado como um array numpy
    """
    
    # carrega com ffmpeg (suporta vários formatos)
    audio, sr = load_audio_use_ffmpeg(audio_path, resample=True, target_sr=target_sr)
    
    # normaliza o áudio
    normalizer = AudioNormalizer()
    audio = normalizer(audio)
    
    return audio


def cinta_audio_input_mapper(ctx, data: Union[str, bytes, np.ndarray, List[str]]) -> MultiModalInputs:
    """mapeia dados de entrada de áudio para o formato vllm multimodalinputs.
    
    esta função está registrada como o mapeador de entrada para o processamento de áudio do cinta.
    ela lida com múltiplos formatos de entrada e os converte em tensores normalizados.
    
    args:
        ctx: contexto do vllm (não utilizado)
        data: dados de entrada de áudio, que podem ser:
            - str: path para um arquivo de áudio
            - bytes: dados de áudio cru em bytes (qualquer formato que o ffmpeg suporte)
            - np.ndarray: forma de onda do áudio pré-carregado
            - list[str]: lista de paths para arquivos de áudio (apenas o primeiro é utilizado)
            
    retorna:
        multimodalinputs contendo:
            - audio: tensor do áudio (float32)
            - audio_length: comprimento do áudio em amostras
    """
    
    # lida com a entrada da lista (pega o primeiro item)
    if isinstance(data, list):
        data = data[0]
        
    audio_waveform = None
    
    if isinstance(data, str):
        # carrega do path do arquivo
        audio_waveform = load_audio(data)
        
    elif isinstance(data, bytes):
        # codifica os bytes diretamente via o ffmpeg stdin pipe para isolar io de temp-file
        audio_waveform, _sr = load_audio_bytes_use_ffmpeg(data, resample=True, target_sr=24000)
        normalizer = AudioNormalizer()
        audio_waveform = normalizer(audio_waveform)
        
    elif isinstance(data, np.ndarray):
        # array do numpy já carregado
        audio_waveform = data
    else:
        raise ValueError(f"tipo de dado de áudio não suportado: {type(data)}")
        
    # converte para um tensor
    audio_tensor = torch.from_numpy(audio_waveform).float()
    audio_length = audio_tensor.shape[0]
    
    return MultiModalInputs({
        "audio": audio_tensor,
        "audio_length": audio_length
    })