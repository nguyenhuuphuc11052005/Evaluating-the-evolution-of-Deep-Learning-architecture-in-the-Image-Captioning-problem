# thuật toán sinh caption bằng greedy hoặc beam search
import torch
from typing import List, Tuple, Optional
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
    Sinh caption bằng greedy search
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
    Giữ lại beam_size giả thuyết tốt nhất ở mỗi bước.
    '''
    def __init__(self, decoder, vocab, max_len = 20, 
                beam_size:int =3, length_penalty_alpha:float = 0.7):
        '''
        Khởi tạo Beam Search
        :param beam_size: số giả thuyết tốt nhất ở mỗi bước để tiếp tục sinh caption
        :param length_penalty_alpha: hệ số length penalty điều chỉnh độ dài câu kết quả 
        '''
        super().__init__(decoder, vocab, max_len)
        self.beam_size = beam_size
        self.length_penalty_alpha = length_penalty_alpha

    def _length_penalty(self, length:int)->float:
        '''
        Sử dụng công thức tính:
        LP(Y) = ((5+|Y|)/6))^alpha với |Y| là độ dài của câu dịch đầu ra
        '''
        return ((5 + length)/6)**self.length_penalty_alpha
    
    def _normalized_score(self, log_prob_sum: float, length:int)->float:
        '''
        Score đã normalize - dùng để so sánh và chọn best sequence
        '''
        return log_prob_sum/ self._length_penalty(length)

    def _get_decoder_input(self, seq: List[int], image_features: torch.Tensor,
                           device: torch.device) -> torch.Tensor:
        '''
        Nếu len(seq) == 1 tức là chỉ chứa <sos> thì dùng image_features
        Các bước sau: embed token cuối cùng trong sequence
        '''
        if len(seq) == 1:
            return image_features # shape (1, 1, embed_size)
        last_token = torch.tensor([seq[-1]], device=device)
        return self.decoder.embed_token(last_token).unsqueeze(1)

    def generate(self, features: torch.Tensor) -> List[int]:
        '''
        Sinh caption từ image feature bằng beam search
        :param features: image feature
        :return: list token ids (không gồm <eos> và <sos>)
        '''
        device = features.device
        # chuẩn hóa shape thành (1, 1, embed_size)
        if features.dim() == 1:
            inputs = features.unsqueeze(0).unsqueeze(0)
        elif features.dim() == 2:
            inputs = features.unsqueeze(1)
        else:
            inputs = features # shape: (1, 1, embed_size)

        # beam chưa seq (List[int]) token ids kể từ <sos>,
        # cumulative_log_prob (float) tổng log_prob tích lũy (chưa normalize)
        # hidden state của decoder (hoặc None)
        active_beams: List[Tuple[List[int], float, Optional[object]]] = [
            ([self.start_idx], 0.0, None)
        ]

        # các sequence đã kết thúc bằng <eos>: (seq, cumulative_log_prob)
        completed: List[Tuple[List[int], float]] = []

        for _ in range(self.max_len):
            if not active_beams:
                # không còn beam đang hoạt động 
                break
            
            candidates: List[Tuple[List[int], float, Optional[object]]] = []
            
            for seq, cum_score, hidden in active_beams:
                dec_input = self._get_decoder_input(seq, inputs, device)

                # decoder 
                outputs, new_hidden = self.decoder.step(dec_input, hidden)

                # log_softmax để lấy log_prob
                log_probs = F.log_softmax(outputs, dim=1) # (1, vocab_size)

                # lấy top beam_size token 
                top_log_probs, top_words = log_probs.topk(self.beam_size, dim=1)

                for i in range(self.beam_size):
                    next_token = top_words[0, i].item()
                    next_cum_score = cum_score + top_log_probs[0, i].item()
                    new_seq = seq + [next_token]

                    candidates.append((new_seq, next_cum_score, new_hidden))

            if not candidates:
                break
            
            # những beam vẫn tiếp tục hoạt động, đưa vào vòng tiếp theo 
            next_active: List[Tuple[List[int], float, Optional[object]]] = []    

            for seq, cum_score, hidden in candidates:
                if seq[-1] == self.end_idx:
                    # sequence hoàn chỉnh thì lưu vào completed
                    completed.append((seq, cum_score))
                else:
                    next_active.append((seq, cum_score, hidden))

            # sắp xếp các active candidates theo normalized score
            # chọn đúng beam_size beams tốt nhất để tiếp tục tìm kiếm
            next_active.sort(
                key=lambda x: self._normalized_score(x[1], len(x[0])),
                reverse=True
            ) 
            active_beams = next_active[:self.beam_size] # lấy beam_size đầu tiên

            # dừng nếu số sequence hoàn chỉnh đã đủ beam_size 
            # và score tốt nhất trong completed >= score tốt nhất trong active
            if len(completed) >= self.beam_size and active_beams:
                # score tốt nhất trong completed
                best_completed_score = max(
                    self._normalized_score(sc, len(s)) for s, sc in completed
                )
                # score tốt nhất trong active
                best_active_score = self._normalized_score(
                    active_beams[0][1], len(active_beams[0][0])
                )
                if best_completed_score >= best_active_score:
                    break

        
        all_candidates:  List[Tuple[List[int], float]] = list(completed)

        # nếu vẫn còn beam chưa kết thúc, đưa vào pool để so sánh
        for seq, cum_score, _ in active_beams:
            all_candidates.append((seq, cum_score))

        if not all_candidates:
            return []
        
        best_seq, _ = max(
            all_candidates, 
            key=lambda x: self._normalized_score(x[1], len(x[0]))
        )

        # bỏ <sos>
        best_seq = best_seq[1:]

        # bỏ <eos> và mọi thứ sau nó
        if self.end_idx in best_seq:
            eos_pos = best_seq.index(self.end_idx)
            best_seq = best_seq[: eos_pos]
        
        return best_seq # token ids