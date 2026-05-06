from typing import List, Tuple, Dict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import re

class ComputeMetrics:
    '''
    Lớp tính các chỉ số đánh giá mô hình 
    - BLEU
    - CIDEr
    - ROUGE
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
    def corpus_bleu_score(self, references: List[List[str]], candidates: List[str])-> float:
        '''
        Tính BLEU ở mức corpus
        :param references: List các list chuỗi tham chiếu
        :param candidate: List chuỗi dữ đoán
        :return: điểm bleu (float)
        '''
        if len(references) != len(candidates):
            raise ValueError("Số lượng references và candidates phải bằng nhau")
        # chuyển thành dạng tokenized
        list_of_refs = []
        hyps = []
        for refs, cand in zip(references, candidates):
            # mỗi câu tham chiếu có thể có nhiểu chuỗi, ta tokenize tất cả
            tokenized_refs = [self.preprocess(r) for r in refs]
            tokenized_cand = self.preprocess(cand)
            list_of_refs.append(tokenized_refs)
            hyps.append(tokenized_cand)
        # smoothing nếu cần 
        if self.smoothing_fn:
            score = corpus_bleu(list_of_refs, hyps, smoothing_function=self.smoothing_fn)
        else:
            score = corpus_bleu(list_of_refs, hyps)
        return score
    
    def sentence_bleu_score(self, references: List[str], candidate: str) -> float:
        '''
        Tính BLEU cho một câu
        :param references: List các câu tham chiếu 
        :param candidate: câu dự đoán
        :return: điểm bleu (float)
        '''
        # tiền xử lý
        refs_tokens = [self.preprocess(r) for r in references]
        hyp_tokens = self.preprocess(candidate)
        if self.smoothing_fn:
            score = sentence_bleu(refs_tokens, hyp_tokens, smoothing_function=self.smoothing_fn)
        else:
            score = sentence_bleu(refs_tokens, hyp_tokens)
    # ROUGE-L
    # CIDEr

    def compute_all(self, references: List[List[str]], candidates: List[str]) -> Dict[str, Dict]:
        '''
        Tính BLEU, ROUGE-L, CIDEr và trả về điểm corpus và điểm câu
        :return: Dict với keys 'BLEU', 'ROUGE-L', 'CIDEr', mỗi key chứa dict {'corpus':.., 'sentence':[]}
        '''
        result = {}
        # BLEU
        bleu_score = self.corpus_bleu_score(references, candidates)
        sent_bleus = [self.sentence_bleu(refs, cand) for refs, cand in zip(references, candidates)]
        result['BLEU'] = {'corpus': bleu_score, 'sentence': sent_bleus}
        # ROUGE-L
        # CIDEr
        return result