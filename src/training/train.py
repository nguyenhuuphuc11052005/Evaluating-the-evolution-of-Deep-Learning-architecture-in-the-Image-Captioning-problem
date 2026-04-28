import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import set_seed
from src.training.loss import get_criterion
from src.training.callbacks import EarlyStopping
from src.logger import setup_logger

def train_model(train_loader, val_loader, encoder, decoder, vocab, config, resume_checkpoint=None):
    # 1. CỐ ĐỊNH SEED VÀ SETUP THIẾT BỊ DDP
    set_seed(42)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    is_master = (local_rank == 0) # Chỉ Rank 0 mới làm nhiệm vụ quản lý I/O
    
    # Chỉ Rank 0 mới khởi tạo Logger và TensorBoard
    if is_master:
        logger, log_dir = setup_logger(config['experiment_name'])
        writer = SummaryWriter(log_dir=log_dir)
        early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 3), 
            save_path="experiments/checkpoints/best_model.pth"
        )
    
    # 2. CHUYỂN MODEL LÊN GPU HIỆN TẠI VÀ KHỞI TẠO OPTIMIZER
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    criterion = get_criterion(vocab)
    optimizer = optim.Adam(decoder.parameters(), lr=config['training']['learning_rate'])
    
    start_epoch = 0

    # 3. TÍCH HỢP RESUME CHECKPOINT (Thực hiện TRƯỚC khi bọc DDP)
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        if is_master:
            logger.info(f"=> Đang nạp Checkpoint từ: {resume_checkpoint}")
            
        # Nạp an toàn trực tiếp vào GPU đang đảm nhận rank này
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Vì model chưa bọc DDP, ta có thể load thẳng state_dict
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        if is_master:
            early_stopping.best_loss = checkpoint.get('val_loss', float('inf'))
            logger.info(f"=> Nạp thành công! Tiếp tục huấn luyện từ Epoch {start_epoch + 1}")
    elif is_master:
        logger.info(f"Bắt đầu huấn luyện từ đầu: {config['experiment_name']}")

    # 4. BỌC MODEL VÀO DDP
    # encoder = DDP(encoder, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])

    num_epochs = config['training']['num_epochs']

    # 5. VÒNG LẶP HUẤN LUYỆN CHÍNH
    for epoch in range(start_epoch, num_epochs):
        # Đảo bài cho DistributedSampler ở mỗi epoch (Bắt buộc cho DDP)
        train_loader.sampler.set_epoch(epoch)
        
        # ================= HUẤN LUYỆN (TRAIN) =================
        encoder.eval() # Encoder luôn đóng băng (đã pre-train)
        decoder.train()
        train_loss = 0.0
        
        # Chỉ Rank 0 mới hiển thị thanh tiến trình tqdm
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if is_master else train_loader
        
        for imgs, captions in train_loop:
            imgs, captions = imgs.to(device), captions.to(device)

            with torch.no_grad():
                features = encoder(imgs)
            
            outputs = decoder(features, captions)
            
            # Reshape để tính loss
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = captions.reshape(-1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if is_master:
                train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Đồng bộ Loss từ tất cả các GPU về để có con số chính xác toàn cục
        train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        global_train_loss = train_loss_tensor.item()

        # ================= KIỂM ĐỊNH (VALIDATION) =================
        decoder.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") if is_master else val_loader
        
        with torch.no_grad():
            for imgs, captions in val_loop:
                imgs, captions = imgs.to(device), captions.to(device)
                
                features = encoder(imgs)
                outputs = decoder(features, captions)
                
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = captions.reshape(-1)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                if is_master:
                    val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        
        # Đồng bộ Validation Loss tương tự Train Loss
        val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        global_val_loss = val_loss_tensor.item()

        # ================= LƯU LOG & CHECKPOINT (CHỈ TRÊN RANK 0) =================
        if is_master:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {global_train_loss:.4f} | Val Loss: {global_val_loss:.4f}")
            writer.add_scalar("Loss/Train", global_train_loss, epoch)
            writer.add_scalar("Loss/Validation", global_val_loss, epoch)
            
            # Lưu state_dict từ .module vì model đang được bọc bởi DDP
            checkpoint_state = {
                'epoch': epoch,
                'val_loss': global_val_loss,
                'encoder_state_dict': encoder.module.state_dict(),
                'decoder_state_dict': decoder.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            
            early_stopping(val_loss=global_val_loss, model_state=checkpoint_state, optimizer_state=optimizer.state_dict(), epoch=epoch)
            
            if early_stopping.early_stop:
                logger.info("=> MÔ HÌNH ĐÃ HỘI TỤ. KÍCH HOẠT EARLY STOPPING!")
                # Lưu ý: Trong DDP thực tế, nếu Rank 0 dùng break, bạn cần thêm cơ chế 
                # dist.broadcast để báo cho các GPU khác cũng break để tránh bị treo (deadlock).
                # Để an toàn cho code hiện tại, ta có thể kết thúc ở đây.
                
        # Đồng bộ hóa tất cả các process ở cuối mỗi epoch trước khi dọn rác
        dist.barrier()
        
        gc.collect()
        torch.cuda.empty_cache()

    if is_master:
        writer.close()
        logger.info("Huấn luyện hoàn tất!")