import os
import torch
import torch.optim as optim
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter

# Tận dụng lại hệ thống Logger từ các mô hình trước
from src.logger import setup_logger 

def train_blip_model(train_loader, val_loader, model, processor, config, logger, writer, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sử dụng AdamW (Biến thể tốt nhất cho Transformer)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    patience = config['training'].get('patience', 3)
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        # ==================== TRAIN LOOP ====================
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train BLIP]")
        
        for batch in train_loop:
            # Đẩy dữ liệu lên GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Hugging Face tự tính CrossEntropy
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # ==================== VALIDATION LOOP ====================
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
        
        # ==================== GHI LOG VÀ TENSORBOARD ====================
        logger.info(f"Epoch [{epoch+1}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        # ==================== EARLY STOPPING & CHECKPOINT ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.info(f"   => [Checkpoint] Val loss đạt đỉnh mới. Đang lưu tại {checkpoint_dir}...")
            
            # Lưu theo chuẩn Hugging Face
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
        else:
            patience_counter += 1
            logger.info(f"   => Val loss không cải thiện ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info("=> KÍCH HOẠT EARLY STOPPING! Kết thúc huấn luyện.")
                break
                
        # Ép dọn rác bộ nhớ để cứu RAM
        gc.collect()
        torch.cuda.empty_cache()

    writer.close()