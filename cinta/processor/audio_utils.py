import os
import threading
import numpy as np
from subprocess import run
from typing import List, Optional, Union, Dict, Any


COMMON_AUDIO_EXTS = [
    '.mp3', '.MP3', '.Mp3', # todas as variações de mp3
    '.m4a', 
    '.mp4', '.MP4',
    '.wav', '.WAV',
    '.m4v',
    '.aac',
    '.ogg',
    '.mov', '.MOV',
    '.opus',
    '.m4b',
    '.flac',
    '.wma', '.WMA',
    '.rm', '.3gp', '.mpeg', '.flv', '.webm', '.mp2', '.aif', '.aiff', '.oga', '.ogv', '.mpga', '.m3u8', '.amr'
]

def load_audio_use_ffmpeg(file: str, resample: bool = False, target_sr: int = 24000):
    """
    abre um arquivo de áudio e lê-lo como forma de onda mono, opcionalmente com reamostragem.
    retorna os dados de áudio e a taxa de amostragem original
    
    parâmetros
    ----------
    file: str
        o arquivo do áudio a ser aberto
    resample: bool
        se deve ou não reamostrar o áudio
    target_sr: int
        a taxa de amostragem alvo, caso seja solicitada uma nova amostragem
    
    retorna
    -------
    um tuple contendo:
    - um array numpy com a forma de onda do áudio em formato float32
    - a taxa de amostragem original do arquivo de áudio
    """
    
    if not resample:
        # primeiro, obtém a taxa de amostragem original
        cmd_probe = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            
            file
        ]
        
        original_sr = int(run(cmd_probe, capture_output=True, check=True).stdout.decode().strip())
    else:
        original_sr = None
        
    # agora carrega o áudio
    sr_to_use = target_sr if resample else original_sr
    
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr_to_use),
        "-"
    ]
    
    out = _run_ffmpeg(cmd).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    return audio_data, sr_to_use


def _get_ffmpeg_max_concurrency() -> int:
    """obtém a concorrência máxima do ffmpeg a partir da variável de ambiente"""
    
    v = os.getenv("CINTA_FFMPEG_MAX_CONCURRENCY", "")
    
    try:
        n = int(v) if v.strip() else 0
    except Exception:
        n = 0
        
    # 0/negativo significa não haver limite explícito
    return n


_FFMPEG_MAX_CONCURRENCY = _get_ffmpeg_max_concurrency()
_FFMPEG_SEM = threading.Semaphore(_FFMPEG_MAX_CONCURRENCY) if _FFMPEG_MAX_CONCURRENCY > 0 else None


def _run_ffmpeg(cmd: list, *, stdin_bytes: bytes = None):
    """roda o ffmpeg com limitação opcional de concorrência global.

    isso é importante para a concorrência de múltiplas requisições do vllm:
    gerar muitos processos do ffmpeg pode saturar a cpu/e/s e causar falhas/tempos limite nas requisições
    """
    
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)


def load_audio_bytes_use_ffmpeg(data: bytes, *, resample: bool = False, target_sr: int = 24000):
    """decodifica bytes de áudio através do pipe stdin do ffmpeg.

    em comparação com a gravação de bytes em um arquivo temporário, isso evita operações de e/s no
    sistema de arquivos e reduz a contenção em situações de alta concorrência de solicitações.
    
    parâmetros
    ----------
    data: bytes
        os bytes de dados de áudio
    resample: bool
        se deve ou não reamostrar o áudio (deve ser true)
    target_sr: int
        a taxa de amostragem alvo, caso seja solicitada uma nova amostragem

    retorna
    -------
    um tuple contendo:
    - um array numpy com a forma de onda do áudio em formato float32
    - a taxa de amostragem
    """
    
    if not resample:
        # para bytes de entrada padrão (stdin), não temos uma maneira barata/robusta de sondar o sr original.
        #
        # mantém o comportamento explícito
        raise ValueError("load_audio_bytes_use_ffmpeg requires resample=True")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-"
    ]
    
    out = _run_ffmpeg(cmd, stdin_bytes=data).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    return audio_data, target_sr


class AudioNormalizer:
    """
    classe de normalização de áudio para o tokenizador cinta.
    
    esta classe fornece normalização de áudio para garantir níveis de entrada consistentes para o tokenizador cinta, mantendo a qualidade do áudio
    """
    
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        """
        inicializa o normalizador de áudio.
        
        args:
            target_db_fs (float): nível alvo de db fs para o áudio. padrão: -25
            eps (float): valor pequeno para evitar divisão por zero. padrão: 1e-6
        """
        
        self.target_dB_FS = target_dB_FS
        self.eps = eps
    
    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        """
        ajusta o áudio para o nível de db fs desejado.
        
        args:
            audio (np.ndarray): sinal de áudio de entrada
            
        retorna:
            tuple: (normalized_audio, rms, scalar)
        """
        
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        normalized_audio = audio * scalar
        
        return normalized_audio, rms, scalar
    
    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        """
        evita recortes reduzindo a escala, se necessário.
        
        args:
            audio (np.ndarray): sinal de áudio de entrada
            scalar (float, optional): fator de escala explícito
            
        retorna:
            tuple: (normalized_audio, scalar)
        """
        
        if scalar is None:
            max_val = np.max(np.abs(audio))
            
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0
        
        return audio / scalar, scalar
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        normaliza o áudio ajustando-o para o nível de db fs desejado e evitando distorções
        
        args:
            audio (np.ndarray): sinal de áudio da entrada
            
        returna:
            np.ndarray: sinal de áudio normalizado
        """
        
        # primeiro, ajusta para o db fs alvo
        audio, _, _ = self.tailor_dB_FS(audio)
        
        # então evita recortes
        audio, _ = self.avoid_clipping(audio)
        
        return audio