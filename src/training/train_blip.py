import os
import torch
import torch.optim as optim
from tqdm import tqdm
import gc

def train_blip_model(train_loader, val_loader, model, processor, config):
    # Khởi tạo GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Sử dụng AdamW (Biến thể tốt nhất cho Transformer) thay vì Adam thường
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Tạo thư mục lưu model
    save_dir = "experiments/checkpoints/blip_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        # --- TRAIN VÒNG LẶP ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train BLIP]")
        
        for batch in train_loop:
            # Đẩy dữ liệu lên GPU (Hugging Face trả về dict, ta duyệt qua dict đó)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # CHỈ CẦN 1 DÒNG NÀY: Truyền ảnh và nhãn vào, mô hình tự tính CrossEntropy
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION VÒNG LẶP ---
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val BLIP]")
        
        with torch.no_grad():
            for batch in val_loop:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"-> Tổng kết Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        
        # --- LƯU CHECKPOINT KIỂU HUGGING FACE ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   [Checkpoint] Val loss giảm. Đang lưu mô hình tại {save_dir}...")
            # Không dùng torch.save nữa, dùng hàm save_pretrained chuẩn của HF
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            
        # Ép dọn rác bộ nhớ để cứu RAM
        gc.collect()
        torch.cuda.empty_cache()

    print("Hoàn tất Fine-tuning BLIP!")