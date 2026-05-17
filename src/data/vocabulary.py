import os
import pickle
import nltk
from collections import Counter
from pycocotools.coco import COCO

# Đảm bảo gói thư viện tách từ đã được tải (rất cần thiết trên Kaggle)
nltk.download('punkt', quiet=True)
# Thêm dòng này để phòng trường hợp lỗi punkt_tab mới nhất của nltk
nltk.download('punkt_tab', quiet=True) 

class Vocabulary(object):
    def __init__(self,
                 vocab_threshold,
                 vocab_file="./vocab.pkl",
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 pad_word="<pad>",  # SỬA Ở ĐÂY: Thêm token pad_word
                 annotations_file="",
                 vocab_from_file=False):
        """
        Khởi tạo bộ từ vựng (Vocabulary).
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word # Lưu pad_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load từ vựng từ file hoặc build mới hoàn toàn."""
        # SỬA Ở ĐÂY: Thay toán tử '&' (bitwise) thành 'and' (logical)
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print(f"-> Đã load Vocabulary thành công từ file: {self.vocab_file}")
        else:
            print("-> Không tìm thấy file (hoặc vocab_from_file=False). Đang build Vocabulary mới...")
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)
            print(f"-> Đã đóng gói và lưu Vocabulary tại: {self.vocab_file}")
        
    def build_vocab(self):
        """Tạo dictionary và nạp các Token đặc biệt vào trước."""
        self.init_vocab()
        
        # SỬA Ở ĐÂY: Phải nạp pad_word vào đầu tiên (thường mang index 0)
        self.add_word(self.pad_word)   # Index 0
        self.add_word(self.start_word) # Index 1
        self.add_word(self.end_word)   # Index 2
        self.add_word(self.unk_word)   # Index 3
        
        # Sau đó mới đọc JSON và nạp các từ bình thường vào
        self.add_captions()

    def init_vocab(self):
        """Khởi tạo dictionary trống."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Thêm một từ vào từ điển nếu nó chưa tồn tại."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Lặp qua toàn bộ captions trong tập train để đếm tần suất."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = list(coco.anns.keys()) # Ép kiểu list để tương thích Python 3
        
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i + 1) % 100000 == 0:
                print("[%d/%d] Đang Tokenize captions..." % (i + 1, len(ids)))

        # Chỉ lấy những từ xuất hiện lớn hơn hoặc bằng vocab_threshold
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for word in words:
            self.add_word(word)

    def __call__(self, word):
        """
        Magic method: Chuyển một từ (string) thành index (integer).
        Sử dụng: vocab("dog") -> trả về index của chữ dog.
        """
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
