import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO

nltk.download('punkt')
nltk.download('punkt_tab') # thêm gói dữ liệu 

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        self.idx = 4

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            # Tokenize câu thành các từ
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            frequencies.update(tokens)

        # Chỉ đưa vào vocab những từ xuất hiện nhiều hơn freq_threshold
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def numericalize(self, text):
        # Chuyển text thành danh sách các index
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]
