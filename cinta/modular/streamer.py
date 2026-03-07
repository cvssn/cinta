from __future__ import annotations
import torch
import asyncio
from queue import Queue
from typing import TYPE_CHECKING, Optional
from transformers.generation import BaseStreamer


class AudioStreamer(BaseStreamer):
    """
    streamer de áudio que armazena trechos de áudio em filas para cada amostra no lote.
    isso permite a geração de áudio em streaming para várias amostras simultaneamente.
    
    parâmetros:
        batch_size (`int`):
            o tamanho do lote para geração
        stop_signal (`any`, *opcional*):
            o sinal a ser inserido na fila quando a geração terminar. o padrão é none.
        timeout (`float`, *opcional*):
            o tempo limite para a fila de áudio. se `none`, a fila ficará bloqueada indefinidamente.
    """
    
    def __init__(
        self, 
        
        batch_size: int,
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None
    ):
        self.batch_size = batch_size
        self.stop_signal = stop_signal
        self.timeout = timeout
        
        # cria uma fila para cada amostra no lote.
        self.audio_queues = [Queue() for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        self.sample_indices_map = {} # mapeia o index de amostra para o index da fila
        
    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """
        recebe trechos de áudio e os coloca nas filas apropriadas.
        
        args:
            audio_chunks: tensor de formato (num_samples, ...) contendo trechos de áudio.
            sample_indices: tensor que indica a qual amostra esses blocos pertencem.
        """
        
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            
            if idx < self.batch_size and not self.finished_flags[idx]:
                # converte para numpy ou manter como tensor, conforme sua preferência
                audio_chunk = audio_chunks[i].detach().cpu()
                
                self.audio_queues[idx].put(audio_chunk, timeout=self.timeout)
    
    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """
        sinaliza o fim da geração para amostras específicas ou para todas as amostras.
        
        args:
            sample_indices: tensor opcional de índices de amostra para finalizar. se none, finaliza tudo.
        """
        
        if sample_indices is None:
            # finaliza todas as amostras
            for idx in range(self.batch_size):
                if not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
        else:
            # finaliza amostras específicas
            for sample_idx in sample_indices:
                idx = sample_idx.item() if torch.is_tensor(sample_idx) else sample_idx
                
                if idx < self.batch_size and not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
    
    def __iter__(self):
        """retorna um iterador sobre o lote de fluxos de áudio."""
        
        return AudioBatchIterator(self)
    
    def get_stream(self, sample_idx: int):
        """obtém o fluxo de áudio para uma amostra específica."""
        
        if sample_idx >= self.batch_size:
            raise ValueError(f"o index de amostra {sample_idx} excede o tamanho do lote {self.batch_size}")
        
        return AudioSampleIterator(self, sample_idx)
    
    
class AudioSampleIterator:
    """iterador para um único fluxo de áudio do lote."""
    
    def __init__(self, streamer: AudioStreamer, sample_idx: int):
        self.streamer = streamer
        self.sample_idx = sample_idx
        
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.streamer.audio_queues[self.sample_idx].get(timeout=self.streamer.timeout)
        
        if value == self.streamer.stop_signal:
            raise StopIteration()
        
        return value
    
    
class AudioBatchIterator:
    """iterador que gera trechos de áudio para todas as amostras no lote."""
    
    def __init__(self, streamer: AudioStreamer):
        self.streamer = streamer
        self.active_samples = set(range(streamer.batch_size))
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.active_samples:
            raise StopIteration()
            
        batch_chunks = {}
        samples_to_remove = set()
        
        # tenta obter fragmentos de todas as amostras ativas
        for idx in self.active_samples:
            try:
                value = self.streamer.audio_queues[idx].get(block=False)
                
                if value == self.streamer.stop_signal:
                    samples_to_remove.add(idx)
                else:
                    batch_chunks[idx] = value
            except:
                # a fila está vazia para esta amostra; ignore-a nesta iteração
                pass
        
        # remove as amostras acabadas
        self.active_samples -= samples_to_remove
        
        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # se nenhum fragmento estiver pronto, mas ainda tivermos amostras
            # ativas, aguarde um pouco e tente novamente
            import time
            
            time.sleep(0.01)
            
            return self.__next__()
        else:
            raise StopIteration()
        
        
class AsyncAudioStreamer(AudioStreamer):
    """
    versão assíncrona do audiostreamer para uso em contextos assíncronos.
    """
    
    def __init__(
        self, 
        
        batch_size: int,
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None
    ):
        super().__init__(batch_size, stop_signal, timeout)
        
        # substitui as filas regulares por filas assíncronas
        self.audio_queues = [asyncio.Queue() for _ in range(batch_size)]
        self.loop = asyncio.get_running_loop()
        
    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """coloca os trechos de áudio nas filas assíncronas apropriadas."""
        
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            
            if idx < self.batch_size and not self.finished_flags[idx]:
                audio_chunk = audio_chunks[i].detach().cpu()
                
                self.loop.call_soon_threadsafe(
                    self.audio_queues[idx].put_nowait, audio_chunk
                )
    
    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """sinaliza o fim da geração para amostras específicas."""
        
        if sample_indices is None:
            indices_to_end = range(self.batch_size)
        else:
            indices_to_end = [s.item() if torch.is_tensor(s) else s for s in sample_indices]
            
        for idx in indices_to_end:
            if idx < self.batch_size and not self.finished_flags[idx]:
                self.loop.call_soon_threadsafe(
                    self.audio_queues[idx].put_nowait, self.stop_signal
                )
                
                self.finished_flags[idx] = True
    
    async def get_stream(self, sample_idx: int):
        """obtém um iterador assíncrono para o fluxo de áudio de uma amostra específica."""
        
        if sample_idx >= self.batch_size:
            raise ValueError(f"index da amostra {sample_idx} excede o tamanho do lote {self.batch_size}")
            
        while True:
            value = await self.audio_queues[sample_idx].get()
            
            if value == self.stop_signal:
                break
            
            yield value
    
    def __aiter__(self):
        """retorna um iterador assíncrono sobre todos os fluxos de áudio."""
        
        return AsyncAudioBatchIterator(self)
    
    
class AsyncAudioBatchIterator:
    """iterador assíncrono para streaming de áudio em lote."""
    
    def __init__(self, streamer: AsyncAudioStreamer):
        self.streamer = streamer
        self.active_samples = set(range(streamer.batch_size))
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self.active_samples:
            raise StopAsyncIteration()
            
        batch_chunks = {}
        samples_to_remove = set()
        
        # cria tarefas para todas as amostras ativas
        tasks = {
            idx: asyncio.create_task(self._get_chunk(idx)) 
            
            for idx in self.active_samples
        }
        
        # aguarda até que pelo menos um bloco esteja pronto
        done, pending = await asyncio.wait(
            tasks.values(), 
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self.streamer.timeout
        )
        
        # cancela as tarefas pendentes
        for task in pending:
            task.cancel()
            
        # processa as tarefas concluídas
        for idx, task in tasks.items():
            if task in done:
                try:
                    value = await task
                    
                    if value == self.streamer.stop_signal:
                        samples_to_remove.add(idx)
                    else:
                        batch_chunks[idx] = value
                except asyncio.CancelledError:
                    pass
                    
        self.active_samples -= samples_to_remove
        
        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # tenta novamente se ainda tivermos amostras ativas.
            return await self.__anext__()
        else:
            raise StopAsyncIteration()
    
    async def _get_chunk(self, idx):
        """função auxiliar para obter um fragmento de uma fila específica."""
        
        return await self.streamer.audio_queues[idx].get()