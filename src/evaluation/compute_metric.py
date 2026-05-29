from typing import List, Dict, Tuple
from rouge_score import rouge_scorer # ROUGE
from pycocoevalcap.cider.cider import Cider # CIDER
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction # BLEU
from nltk.translate.meteor_score import meteor_score # METEOR
import re
import numpy as np

import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class ComputeMetrics:
    '''
    Lớp tính các chỉ số đánh giá mô hình 
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - CIDEr
    - ROUGE-L
    - METEOR
    '''
    def __init__(self, smooth: bool = True, rouge:str='rougeL'):
        '''
        Khởi tạo ComputeMetric
        :param smooth: giúp kết quả có ý nghĩa hơn mặc dù p_n=0 (không match từ nào với n-grams - n>1)
            (ở đây dùng method4: geometric smoothing, nếu p_n = 0, 
            nó sẽ ước lượng giá trị dựa trên xu hướng giảm dần của n_grams trước đó)
        :param rouge: mặc định dùng Rouge-L (chuỗi con chung dài nhất), có thể thay bằng 'rouge1',...
            (tham số mặc định: use_stemmer=True để đưa về từ gốc)
        '''
        self.smooth = smooth 
        if smooth:
            self.smoothing_fn = SmoothingFunction().method4
        else:
            self.smoothing_fn = None
        # khởi tạo rouge metric 
        self.rouge = rouge 
        self.rouge_scorer = rouge_scorer.RougeScorer([self.rouge], use_stemmer=True)
        # khởi tạo CIDER
        self.cider_scorer = Cider() 

    def _preprocess(self, text: str) -> List[str]:
        '''
        Tiền xử lý một câu: lowercase, loại bỏ punctation 
        :param text: (str) chuỗi đầu vào (mặc định đã loại các token vô nghĩa: eos, sos, pad)
        :return: danh sách tokens
        '''
        # SỬA: dùng tokenizer của ntlk để tránh xóa "don't" -> "dont"
        return nltk.word_tokenize(text.lower())
    
    # BLEU
    def _corpus_bleu_score(self, references: List[List[str]], 
                          hypotheses: List[str], weights: Tuple)-> float:
        '''
        Tính BLEU ở mức corpus
        :param references: List các list chuỗi tham chiếu
        :param hypotheses: List chuỗi dữ đoán
        :return: điểm bleu (float)
        '''
        if len(references) != len(hypotheses):
            raise ValueError("Số lượng references và hypotheses phải bằng nhau")
        
        # Tránh BLEU crash nếu hypotheses rỗng 
        hypotheses = [
            hyp if len(hyp) > 0 else ''
            for hyp in hypotheses
        ]
        # chuyển thành dạng tokenized
        list_of_refs = []
        hyps = []
        for refs, cand in zip(references, hypotheses):
            # mỗi câu tham chiếu có thể có nhiểu chuỗi, ta tokenize tất cả
            tokenized_refs = [self._preprocess(r) for r in refs]
            tokenized_cand = self._preprocess(cand)
            list_of_refs.append(tokenized_refs)
            hyps.append(tokenized_cand)
        # smoothing nếu cần 
        if self.smoothing_fn:
            score = corpus_bleu(list_of_refs, hyps,
                                weights=weights, smoothing_function=self.smoothing_fn)
        else:
            score = corpus_bleu(list_of_refs, hyps, weights=weights)
        return score
    
    # ROUGE-L
    def _rouge_l_score(self, references: List[List[str]], hypotheses: List[str]) -> float:
        '''
        Tính ROUGE-L dựa trên chuỗi chung dài nhất (longest common subsequence)
        :param references: chuỗi tham chiếu (có thể có nhiều tham chiếu cho mỗi bức ảnh)
        :param hypotheses: chuỗi dự đoán tốt nhất cho từng bức ảnh
        :return: điểm số của rouge metric tính bằng f1 (kết hợp giữa precision và recall)
        '''
        if len(references) != len(hypotheses):
            raise ValueError("Số lượng references và hypotheses phải bằng nhau")
        
        best_scores = [] # lưu điểm số cao nhất 

        for hypothesis, reference in zip(hypotheses, references):
            # lưu điểm cao nhất khi so sánh câu pred với từng câu tham chiếu
            # ví dụ: score(hyp, ref1) = 0.5, score(hyp, ref2) = 0.8
            # thì score(hyp, ref) = 0.8 
            # với ý tưởng: chỉ cần sinh caption đúng với 1 trong những truth ground là được
            best_score = 0.0 

            for ref in reference:
                scores = self.rouge_scorer.score(hypothesis, ref)

                # lấy f1 để cân bằng precision và recall
                f1 = scores[self.rouge].fmeasure # SỬA: đổi tên biến động 

                # lấy điểm số cao hơn 
                if f1 > best_score:
                    best_score = f1

            # thêm vào danh sách 
            best_scores.append(best_score)

        # lấy điểm trung bình làm tròn 5 đơn vị
        score = round(float(np.mean(best_scores)), 5) if best_scores else 0.0    
        return score

    # CIDEr
    def _cider_score(self, references: List[List[str]], hypotheses: List[str]) -> float:
        '''
        Tính CIDEr(Consensus-based Image Description Evaluation)
            (Ý tưởng: sử dụng kỹ thuật TF-IDF để gán trọng số cho các n_gram,
            giúp ưu tiên các từ mang thông tin quan trọng và giảm nhẹ ảnh hưởng của các từ phổ biến
            Dùng cosine similarity để đánh giá sự tương đồng giữa các vector n_gram)
        :param references: chuỗi tham chiếu (có thể có nhiều tham chiếu cho mỗi bức ảnh)
        :param hypotheses: chuỗi dự đoán tốt nhất cho từng bức ảnh
        :return: điểm số của cide-r
        '''
        if len(references) != len(hypotheses):
            raise ValueError("Số lượng references và hypotheses phải bằng nhau")
        
        if self.cider_scorer is None:
            print("Chưa khởi tạo CIDER!")
            return 0.0
        # chuẩn bị dữ liệu cho phù hợp với cider
        gts = {
            i: [' '.join(self._preprocess(r)) for r in refs]
            for i, refs in enumerate(references)
        }
        res = {
            i: [' '.join(self._preprocess(h))]
            for i, h in enumerate(hypotheses)
        }
        # tính toán 
        score, _ = self.cider_scorer.compute_score(gts, res)
        return round(score,5)

    # METEOR
    def _meteor_score(self, references: List[List[str]], hypotheses: List[str]) -> float:
        '''
        Tính METEOR:
            (Ý tưởng: thực hiện so khớp từ vựng dựa trên các dạng:
            khớp chính xác, khớp theo gốc từ (stemming) và khớp đồng nghĩa.
            Sau đó, nó áp dụng một hệ số phạt phân mảnh (fragmentation penalty) 
            để đánh giá độ trôi chảy và cấu trúc thứ tự từ trong câu)
        :param references: chuỗi tham chiếu (có thể có nhiều tham chiếu cho mỗi bức ảnh)
        :param hypotheses: chuỗi dự đoán tốt nhất cho từng bức ảnh
        :return: điểm số của meteor
        '''
        if len(references) != len(hypotheses):
            raise ValueError("Số lượng references và hypotheses phải bằng nhau")
        
        # lưu điểm của từng câu
        sentence_scores = []

        for refs, hyp in zip(references, hypotheses):
            # tiền xử lý đầu vào vì meteor cần có dạng token list
            # references=[['this','is','cat'],
            #           ['this','is','a','cat']],
            # hypothesis=['this','is','cat']

            hyp_tokens = self._preprocess(hyp)
            ref_tokens_list = [self._preprocess(ref) for ref in refs]

            if not hyp_tokens:
                score = 0.0
            else:
                score = meteor_score(ref_tokens_list, hyp_tokens)

            sentence_scores.append(score)

        # tính điểm cho corpus
        corpus = round(float(np.mean(sentence_scores)), 5) if sentence_scores else 0.0
        return corpus

    def compute_all(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, Dict]:
        '''
        Tính BLEU, ROUGE-L, CIDEr, METEOR và trả về điểm corpus và điểm câu
        :return: Dict với keys 'BLEU', 'ROUGE-L', 'CIDEr', 'METEOR' với values là score tương ứng
        '''
        results = {}
        # BLEU-1
        bleu1 = self._corpus_bleu_score(references, hypotheses, weights=(1.0, 0, 0, 0))
        
        # BLEU-2
        bleu2 = self._corpus_bleu_score(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        
        # BLEU-3
        bleu3 = self._corpus_bleu_score(references, hypotheses, weights=(1/3, 1/3, 1/3, 0))
        
        # BLEU-4
        bleu4 = self._corpus_bleu_score(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        
        rougeL = self._rouge_l_score(references=references, hypotheses=hypotheses)
        cider = self._cider_score(references=references, hypotheses=hypotheses)
        meteor = self._meteor_score(references=references, hypotheses=hypotheses)

        # cập nhật các metric
        # BLEU
        results['BLEU'] = {
            'BLEU-1': round(bleu1,5),
            'BLEU-2': round(bleu2,5),
            'BLEU-3': round(bleu3,5),
            'BLEU-4': round(bleu4,5)
        }
        # ROUGE-L
        results['ROUGE_L'] = rougeL
        # CIDEr
        results['CIDEr'] = cider
        # METEOR
        results['METEOR'] = meteor
        return results
    
    def compute_single_image(self, references: List[str], hypothesis: str) -> Dict:
        '''
        Tính metric: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L cho mỗi ảnh
        :param references: chuỗi tham chiếu (có thể có nhiều tham chiếu cho mỗi bức ảnh)
        :param hypotheses: chuỗi dự đoán tốt nhất cho từng bức ảnh
        :return: scores
        '''
        hyp = self._preprocess(hypothesis)
        refs = [self._preprocess(ref) for ref in references]

        # BLEU
        bleu1 = sentence_bleu(references=refs, hypothesis=hyp, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu(references=refs, hypothesis=hyp, weights=(0.5, 0.5, 0, 0))
        bleu3 = sentence_bleu(references=refs, hypothesis=hyp, weights=(1/3, 1/3, 1/3, 0))
        bleu4 = sentence_bleu(references=refs, hypothesis=hyp, weights=(0.25, 0.25, 0.25, 0.25))
    
        # METEOR
        meteor = meteor_score(references=refs, hypothesis=hyp)    
        # ROUGE-L
        best_rouge = 0.0
        # dùng chuỗi gốc vì rouge cần dạng [str] chứ không dùng với [token1, token2,..]
        for ref in references:
            scores = self.rouge_scorer.score(ref, hypothesis)
            f1 = scores[self.rouge].fmeasure
            best_rouge = max(best_rouge, f1)
            
        avg_score = round(float((bleu4 + meteor + best_rouge) / 3),5)
        results = dict()
        results['BLEU'] = {
            'BLEU-1': round(bleu1,5),
            'BLEU-2': round(bleu2,5),
            'BLEU-3': round(bleu3,5),
            'BLEU-4': round(bleu4,5)
        }
        # ROUGE-L
        results['ROUGE_L'] = round(best_rouge,5)
        # METEOR
        results['METEOR'] = round(meteor,5)
        results['AVERAGE'] = round(avg_score,5)
        return results
        
def get_eval_score(references: List[List[str]], hypotheses: List[str], 
                   smooth:bool=True, rouge:str='rougeL') -> Dict: 
    '''
    Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    :param references: chuỗi tham chiếu (có thể có nhiều tham chiếu cho mỗi bức ảnh)
    :param hypotheses: chuỗi dự đoán tốt nhất cho từng bức ảnh
    :return: scores
    '''
    evaluator = ComputeMetrics(smooth=smooth, rouge=rouge)
    return evaluator.compute_all(references, hypotheses)

def get_eval_image(references: List[str], hypothesis: str, 
                   smooth:bool=True, rouge:str='rougeL') -> Dict: 
    '''
    Calculate BLEU1~4, METEOR, ROUGE_L scores for one image.
    :param references: list caption tham chiếu của một ảnh
    :param hypothesis: caption dự đoán của mô hình cho ảnh đó
    :return: scores
    '''
    evaluator = ComputeMetrics(smooth=smooth, rouge=rouge)
    return evaluator.compute_single_image(references, hypothesis)
