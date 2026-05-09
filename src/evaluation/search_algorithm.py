# thuật toán sinh caption bằng greedy hoặc beam search
import torch
from typing import List
import torch.nn.functional as F
from abc import ABC, abstractmethod
class BaseSearch(ABC):
    '''
    Lớp base cho các thuật toán tìm kiếm
    '''

    def __init__(self, decoder, vocab, max_len: int = 20):
        '''
        Khởi tạo lớp base
        :param decoder: model decoder
        :param vocab: vocabulary 
        :param max_len: độ dài chuỗi tối đa (mặc định là 20)
        '''
        super().__init__()
        self.decoder = decoder
        self.vocab = vocab
        self.max_len = max_len
        self.start_idx = vocab.stoi["<sos>"]
        self.end_idx = vocab.stoi["<eos>"]
        self.pad_idx = vocab.stoi["<pad>"]

    
    @abstractmethod
    def generate(self, features: torch.Tensor) -> List[int]:
        '''
        Sinh token ids từ feature ảnh
        :param features: vector đặc trưng của ảnh
        :return: token ids list
        '''
        raise NotImplementedError


class GreedySearch(BaseSearch):
    '''
    Sinh caption theo greedy search
    tại mỗi bước chọn từ có xác suất lớn nhất
    '''
    def __init__(self, decoder, vocab, max_len = 20):
        super().__init__(decoder, vocab, max_len)

    def generate(self, features: torch.Tensor) -> List[int]:
        sampled_ids = []
        batch_size = features.size(0)
        
        # Đảm bảo input đầu tiên có shape (batch_size, 1, embed_size)
        # Nếu features là (batch_size, embed_size) -> unsqueeze(1)
        if features.dim() == 2:
            inputs = features.unsqueeze(1)
        else:
            inputs = features

        # note: có thể dùng khởi tạo state của model
        states = None 
        
        for _ in range(self.max_len):
            # predict token tiếp theo
            outputs, states = self.decoder.step(inputs, states)       
            
            # Chọn ra từ có xác suất cao nhất
            predicted = outputs.argmax(dim=1)
            
            sampled_ids.append(predicted.item())

            # Nếu gặp <eos> thì dừng 
            if predicted.item() == self.end_idx:
                break
            
            # Đưa từ vừa dự đoán làm input cho bước tiếp theo
            inputs = self.decoder.embed_token(predicted).unsqueeze(1)
            
        return sampled_ids

# có thể thêm thuật toán Beam Search với ý tưởng giữ lại k từ tốt nhất để tiếp tục sinh từ tiếp theo
class BeamSearch(BaseSearch):
    '''
    Sinh caption bằng beam search
    giữ lại beam_size giả thuyết tốt nhất ở mỗi bước
    '''
    pass