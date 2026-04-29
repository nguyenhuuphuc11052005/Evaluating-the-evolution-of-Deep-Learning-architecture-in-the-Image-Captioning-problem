import os
import torch.cuda.amp as amp # Thêm thư viện
import torch.nn as nn
from logging import config
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.logger import setup_logger
from src.utils import set_seed
from src.training.loss import get_criterion
from src.training.callbacks import EarlyStopping
import gc

def train_model(train_loader, val_loader, encoder, decoder, vocab,config,resume_checkpoint=None):
    # 1. Setup ban đầu và Cố định Seed
    set_seed(42) # Đảm bảo reproducibility 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = encoder.to(device), decoder.to(device)
    
    scaler = amp.GradScaler() # Khởi tạo GradScaler cho Mixed Precision Training
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    exp_name = config['experiment_name']

    # ================= CẬP NHẬT ĐƯỜNG DẪN ĐỘNG TẠI ĐÂY =================
    # 1. Đường dẫn lưu Checkpoint cho riêng mô hình này
    checkpoint_dir = os.path.join("experiments/checkpoints", exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True) # Tự động tạo thư mục nếu chưa có
    
    # 2. Đường dẫn lưu log TensorBoard cho riêng mô hình này
    tb_log_dir = os.path.join("experiments/logs", exp_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
    # 2. Khởi tạo Loss, Optimizer và Callbacks
    criterion = get_criterion(vocab)
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate) # Tối ưu hóa Adam 
    early_stopping = EarlyStopping(patience=3, save_path=model_save_path)
    
    # 3. Quản lý thực nghiệm với TensorBoard
    # Ghi log biểu đồ tự động 
    logger, log_dir = setup_logger(exp_name)
    if torch.cuda.device_count() > 1:
        logger.info(f"Kích hoạt chạy song song trên {torch.cuda.device_count()} GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    
    # 4. Xử lý Checkpoint nếu có (Tiếp tục huấn luyện từ checkpoint)
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        logger.info(f"=> Đang nạp Checkpoint từ: {resume_checkpoint}")
        # Map_location giúp nạp an toàn kể cả khi train ở GPU nhưng load ở CPU
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Nạp trạng thái cho Encoder/Decoder (Xử lý cẩn thận vụ DataParallel)
        if isinstance(encoder, nn.DataParallel):
            encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
        if isinstance(decoder, nn.DataParallel):
            decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
        # Nạp trạng thái cho Optimizer (Giữ lại gia tốc học của Adam)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Cập nhật epoch bắt đầu và loss tốt nhất
        start_epoch = checkpoint['epoch'] + 1
        early_stopping.best_loss = checkpoint['val_loss']
        
        logger.info(f"=> Đã nạp thành công! Tiếp tục huấn luyện từ Epoch {start_epoch + 1}")
    else:
        logger.info(f"Bắt đầu huấn luyện từ đầu: {config['experiment_name']}")
        logger.info(f"Cấu hình: {config}")
    
    for epoch in range(num_epochs):
        # ================= HUẤN LUYỆN (TRAIN) =================
        encoder.eval() # Pre-trained Encoder luôn đóng băng
        decoder.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for imgs, captions in train_loop:
            imgs, captions = imgs.to(device), captions.to(device)

            max_len = config['model'].get('max_seq_len', 100)
            if captions.size(1) > max_len:
                captions = captions[:, :max_len]# Cắt bỏ những từ vượt quá giới hạn

            # Trích xuất đặc trưng (Không tính gradient cho ảnh)
            with amp.autocast(): 
                with torch.no_grad():
                    features = encoder(imgs)
                outputs = decoder(features, captions)
                
                # --- FIX LỖI SEQUENCE LENGTH (CHO TRAIN) ---
                if outputs.size(1) < captions.size(1):
                    # Transformer: Dịch target sang phải, ép bộ nhớ liên tục
                    targets = captions[:, 1:].contiguous().view(-1)
                else:
                    # LSTM: Giữ nguyên
                    targets = captions.contiguous().view(-1)

                # Ép outputs liên tục và đổi shape
                outputs = outputs.contiguous().view(-1, outputs.size(-1))

                loss = criterion(outputs, targets)

            # Backward pass và Cập nhật trọng số 
            
            scaler.scale(loss).backward()            
            scaler.unscale_(optimizer)# Clip gradient tránh nổ đạo hàm cho LSTM
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch) # Lưu log lên TensorBoard 

        # ================= KIỂM ĐỊNH (VALIDATION) =================
        decoder.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for imgs, captions in val_loop:
                imgs, captions = imgs.to(device), captions.to(device)
                
                max_len = config['model'].get('max_seq_len', 100)
                if captions.size(1) > max_len:
                    captions = captions[:, :max_len]# Cắt bỏ những từ vượt quá giới hạn

                features = encoder(imgs)
                outputs = decoder(features, captions)
                
                if outputs.size(1) < captions.size(1):
                    # Dành cho Transformer: Dịch target sang phải 1 bước (bỏ <sos>)
                    targets = captions[:, 1:].reshape(-1)
                else:
                    # Dành cho LSTM: Giữ nguyên do vector ảnh đã bù vào độ dài
                    targets = captions.reshape(-1)

                outputs = outputs.reshape(-1, outputs.shape[-1])
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch) # Lưu log Val lên TensorBoard 

        logger.info(f"Epoch [{epoch+1}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        encoder_state = encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict()
        decoder_state = decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict()
        # ================= XỬ LÝ CHECKPOINT & EARLY STOPPING =================
        model_state = {
            'encoder': encoder_state,
            'decoder': decoder_state
        }
        
        # Đưa vào EarlyStopping để tự đánh giá và lưu checkpoint 
        early_stopping(
            val_loss=avg_val_loss, 
            model_state=model_state, 
            optimizer_state=optimizer.state_dict(), 
            epoch=epoch
        )
        
        if early_stopping.early_stop:
            logger.info("=> MÔ HÌNH ĐÃ HỘI TỤ. KÍCH HOẠT EARLY STOPPING!")
            break
        # Dọn dẹp bộ nhớ GPU sau mỗi epoch
        gc.collect()
        torch.cuda.empty_cache()
        

    writer.close()
    logger.info("Huấn luyện hoàn tất!")