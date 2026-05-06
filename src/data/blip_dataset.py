import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class BlipCocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, processor, max_target_length=50):
        """
        Args:
            root_dir (str): Thư mục chứa ảnh.
            ann_file (str): Đường dẫn file JSON chứa annotations.
            processor: BlipProcessor từ Hugging Face.
            max_target_length (int): Độ dài tối đa của câu caption.
        """
        self.root_dir = root_dir
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
            
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Tùy thuộc vào format JSON của bạn, key có thể là 'file_name' hoặc 'image_id'
        img_name = ann.get('file_name', f"{ann.get('image_id'):012d}.jpg")
        img_path = os.path.join(self.root_dir, img_name)

        # Mở ảnh và dọn rác ngay bằng 'with' (Kỹ thuật cứu RAM đã học ở vòng trước)
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        caption = ann['caption']

        # --- ĐIỂM SÁNG GIÁ NHẤT CỦA BLIP ---
        # Processor sẽ tự động: 
        # 1. Resize, Normalize ảnh cho Vision Transformer (ViT)
        # 2. Tokenize, Thêm <bos>, <eos>, và Pad câu caption
        encoding = self.processor(
            images=image, 
            text=caption, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True, 
            return_tensors="pt"
        )

        # Processor trả về tensor có batch_dim = 1 (Ví dụ: [1, 3, 384, 384]).
        # Ta cần squeeze(0) để loại bỏ số 1 này, DataLoader sẽ tự gộp batch sau.
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # --- BÍ QUYẾT FINE-TUNING CHUẨN CÔNG NGHIỆP ---
        # Đối với bài toán sinh văn bản, ta sao chép 'input_ids' làm 'labels' (nhãn)
        item["labels"] = item["input_ids"].clone()
        
        # Hàm Loss của Hugging Face mặc định sẽ BỎ QUA các token có id = -100
        # Do đó ta thay thế toàn bộ id của token <pad> thành -100 để mạng không bị nhiễu
        pad_token_id = self.processor.tokenizer.pad_token_id
        item["labels"][item["labels"] == pad_token_id] = -100

        return item

def get_blip_loader(root_dir, ann_file, processor, batch_size=16, is_train=True):
    """
    Hàm tiện ích để tạo DataLoader. 
    Lưu ý: Không cần hàm collate_fn tự viết nữa, PyTorch tự gộp các dict rất mượt.
    """
    dataset = BlipCocoDataset(root_dir, ann_file, processor)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,      # Giữ nguyên 0 để an toàn RAM trên Kaggle
        pin_memory=False    # Tắt để giữ mát RAM
    )
    return loader