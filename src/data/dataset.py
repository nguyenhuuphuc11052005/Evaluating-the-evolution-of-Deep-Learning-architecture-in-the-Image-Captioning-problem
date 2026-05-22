import os
import torch
import nltk # Thêm nltk để tokenize câu
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, vocab, transform=None, limit=50000):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.vocab = vocab
        self.transform = transform
        
        # Lấy tất cả ID của caption, sau đó giới hạn ở con số 50,000
        self.ids = list(self.coco.anns.keys())[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = str(self.coco.anns[ann_id]['caption']) # Ép kiểu string để an toàn
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Load ảnh
        img = Image.open(os.path.join(self.root_dir, path)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # --- SỬA Ở ĐÂY: Tokenize và Numericalize theo chuẩn Vocabulary mới ---
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        
        # Bắt đầu bằng token <start>
        numericalized_caption = [self.vocab.word2idx[self.vocab.start_word]]
        
        # Chuyển đổi từng từ thành index thông qua __call__ của vocab
        numericalized_caption.extend([self.vocab(token) for token in tokens])
        
        # Kết thúc bằng token <end>
        numericalized_caption.append(self.vocab.word2idx[self.vocab.end_word])

        return img, torch.tensor(numericalized_caption)

# Transform cơ bản cho Transfer Learning 
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Chuẩn ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

class CapsCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Batch là danh sách các tuple (image, caption)
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        # Pad sequence để các câu bằng nhau
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(root_dir, ann_file, vocab, transform, batch_size=32, num_workers=0, limit=50000, shuffle=True):
    dataset = CocoDataset(root_dir, ann_file, vocab, transform, limit)
    
    # --- SỬA Ở ĐÂY: Xử lý pad_idx ---
    # File Vocabulary của bạn dùng word2idx. Nếu bạn chưa add("<pad>") vào vocab, 
    # hàm này sẽ tự động fallback về mượn index 0 (của <start>) làm pad value để tránh lỗi.
    if "<pad>" in vocab.word2idx:
        pad_idx = vocab.word2idx["<pad>"]
    else:
        pad_idx = 0 

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle, # Trộn ảnh lên cho mỗi epoch
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )
    return loader
