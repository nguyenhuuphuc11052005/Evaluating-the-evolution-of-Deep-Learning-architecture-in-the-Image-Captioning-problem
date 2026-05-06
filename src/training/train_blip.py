import os
import torch
import torch.optim as optim
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter

# Tận dụng lại hệ thống Logger từ các mô hình trước
from src.logger import setup_logger 

def train_blip_model(train_loader, val_loader, model, processor, config):
    # 1. Khởi tạo Hệ thống Logging & TensorBoard
    # Tự động tạo thư mục log dựa trên tên experiment trong file yaml
    logger, log_dir = setup_logger(config['experiment_name'])
    writer = SummaryWriter(log_dir=log_dir)
    
    logger.info(f"=== BẮT ĐẦU HUẤN LUYỆN BLIP: {config['experiment_name']} ===")

    # 2. Khởi tạo thiết bị và Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Thư mục lưu Checkpoint được đặt tên theo experiment
    save_dir = os.path.join("experiments/checkpoints", config['experiment_name'])
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    # 3. Vòng lặp huấn luyện
    for epoch in range(config['training']['num_epochs']):
        # --- VÒNG LẶP TRAIN ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train BLIP]")
        
        for batch in train_loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VÒNG LẶP VALIDATION ---
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
        
        # --- GHI LOG VÀ TENSORBOARD (Thay thế print) ---
        logger.info(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        # --- LƯU CHECKPOINT ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"=> [Checkpoint] Validation Loss đạt kỷ lục mới! Đang lưu mô hình tại: {save_dir}")
            
            # Lưu trọng số mô hình và file cấu hình Tokenizer
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            
        # Dọn rác bộ nhớ Kaggle
        gc.collect()
        torch.cuda.empty_cache()

    # Đóng luồng ghi
    logger.info("Hoàn tất quá trình Fine-tuning BLIP!")
    writer.close()