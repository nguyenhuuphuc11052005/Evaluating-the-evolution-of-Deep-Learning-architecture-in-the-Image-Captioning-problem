import torch.nn as nn
import torch
from typing import Tuple
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module, ABC):
    '''
    Lớp base cho decoder, chuẩn hóa interface cho mọi decoder   
    '''
    def __init__(self):
        '''
        Khởi tạo BaseDecoder
        '''
        super(BaseDecoder, self).__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        '''
        Train steps
        :param features: vector đặc trưng ảnh từ encoder - shape: (batch_size, num_regions, embed_size)
        :param captions: caption ground truth dùng để teacher forcing - shape: (batch_size, seq_len)
        :return: logits dự đoán cho từng từ - shape: (batch_size, seq_len, vocab_size)
        '''
        raise NotImplementedError
    
    @abstractmethod
    def step(self, inputs: torch.Tensor, states=None) -> Tuple[torch.Tensor, tuple]:
        '''
        Dự đoán token tiếp theo 
        :param inputs: token embedding hiện tại 
        :param states: hidden states
        :return: logits dự đoán từ tiếp theo
        '''
        raise NotImplementedError
    
    @abstractmethod
    def embed_token(self, tokens: torch.Tensor) -> torch.Tensor:
        '''
        Chuyển token ids thành embeddings
        :param tokens: tokens
        :return: embedding 
        '''
        pass