import os
import torch
import torch.optim as optim
from tqdm import tqdm
import gc

def train_blip_model(train_loader, val_loader, model, processor, config, logger, writer, checkpoint_dir, accelerator):
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- PHÉP THUẬT CỦA ACCELERATE ---
    # Tự động chia batch, chia model, đồng bộ GPU mà không cần viết code DDP
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    best_val_loss = float('inf')
    patience = config['training'].get('patience', 3)
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        # ==================== TRAIN LOOP ====================
        model.train()
        train_loss = 0.0
        
        # Chỉ hiển thị thanh tiến trình trên GPU chính cho đỡ rối
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=not accelerator.is_local_main_process)
        
        for batch in train_loop:
            # KHÔNG CẦN DÒNG NÀY NỮA (accelerate đã tự làm): 
            # batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            
            # Loss bây giờ đã là 1 số chuẩn, không bị mảng nhiều chiều như DP
            loss = outputs.loss
            
            # THAY THẾ loss.backward() BẰNG:
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # ==================== VALIDATION LOOP ====================
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=not accelerator.is_local_main_process)
        
        with torch.no_grad():
            for batch in val_loop:
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # ==================== GHI LOG VÀ CHECKPOINT ====================
        # Đợi cả 2 GPU chạy xong epoch mới ghi log
        accelerator.wait_for_everyone() 
        
        if accelerator.is_local_main_process:
            logger.info(f"Epoch [{epoch+1}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logger.info(f"   => [Checkpoint] Val loss đạt đỉnh mới. Đang lưu...")
                
                # Bóc model ra khỏi lớp vỏ bọc song song để lưu chuẩn định dạng
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
            else:
                patience_counter += 1
                logger.info(f"   => Val loss không cải thiện ({patience_counter}/{patience})")
                
        # Đồng bộ lệnh Early Stopping cho 2 GPU
        if patience_counter >= patience:
            if accelerator.is_local_main_process:
                logger.info("=> KÍCH HOẠT EARLY STOPPING!")
            break
                
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_local_main_process:
        writer.close()