from typing import List, Tuple, Dict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import re

class ComputeMetrics:
    '''
    Lớp tính các chỉ số đánh giá mô hình 
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - CIDEr
    - ROUGE-L
    - METOR
    '''
    def __init__(self, smooth: bool = True):
        self.smooth = smooth 
        if smooth:
            self.smoothing_fn = SmoothingFunction().method1
        else:
            self.smoothing_fn = None

    def preprocess(self, text: str) -> List[str]:
        '''
        Tiền xử lý một câu: lowercase, loại bỏ punctation 
        :param text: (str) chuỗi đầu vào
        :return: danh sách tokens
        '''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()

        return tokens
    # BLEU
    def corpus_bleu_score(self, references: List[List[str]], 
                          hypotheses: List[str], weights: tuple)-> float:
        '''
        Tính BLEU ở mức corpus
        :param references: List các list chuỗi tham chiếu
        :param candidate: List chuỗi dữ đoán
        :return: điểm bleu (float)
        '''
        if len(references) != len(hypotheses):
            raise ValueError("Số lượng references và candidates phải bằng nhau")
        
        # Tránh BLEU crash nếu hypotheses rỗng 
        hypotheses = [
            hyp if len(hyp) > 0 else ['']
            for hyp in hypotheses
        ]
        # chuyển thành dạng tokenized
        list_of_refs = []
        hyps = []
        for refs, cand in zip(references, hypotheses):
            # mỗi câu tham chiếu có thể có nhiểu chuỗi, ta tokenize tất cả
            tokenized_refs = [self.preprocess(r) for r in refs]
            tokenized_cand = self.preprocess(cand)
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
    # CIDEr

    def compute_all(self, references: List[List[str]], candidates: List[str]) -> Dict[str, Dict]:
        '''
        Tính BLEU, ROUGE-L, CIDEr và trả về điểm corpus và điểm câu
        :return: Dict với keys 'BLEU', 'ROUGE-L', 'CIDEr', mỗi key chứa dict {'corpus':.., 'sentence':[]}
        '''
        results = {}
        # BLEU-1
        bleu1 = self.corpus_bleu_score(references, candidates, weights=(1.0, 0, 0, 0))
        
        # BLEU-2
        bleu2 = self.corpus_bleu_score(references, candidates, weights=(0.5, 0.5, 0, 0))
        
        # BLEU-3
        bleu3 = self.corpus_bleu_score(references, candidates, weights=(1/3, 1/3, 1/3, 0))
        
        # BLEU-4
        bleu4 = self.corpus_bleu_score(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
        
        
        results['BLEU'] = {

            'BLEU-1': {
                'corpus': bleu1,
            },

            'BLEU-2': {
                'corpus': bleu2,
            },

            'BLEU-3': {
                'corpus': bleu3,
            },

            'BLEU-4': {
                'corpus': bleu4,
            }
        }
        # ROUGE-L
        # CIDEr
        return results
