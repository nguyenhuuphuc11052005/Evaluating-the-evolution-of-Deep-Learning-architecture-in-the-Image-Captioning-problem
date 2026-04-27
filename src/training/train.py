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

def train_model(train_loader, val_loader, encoder, decoder, vocab,config):
    # 1. Setup ban đầu và Cố định Seed
    set_seed(42) # Đảm bảo reproducibility 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = encoder.to(device), decoder.to(device)
    

    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    # 2. Khởi tạo Loss, Optimizer và Callbacks
    criterion = get_criterion(vocab)
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate) # Tối ưu hóa Adam 
    early_stopping = EarlyStopping(patience=3, save_path="experiments/checkpoints")
    
    # 3. Quản lý thực nghiệm với TensorBoard
    # Ghi log biểu đồ tự động 
    logger, log_dir = setup_logger(config['experiment_name'])
    if torch.cuda.device_count() > 1:
        logger.info(f"Kích hoạt chạy song song trên {torch.cuda.device_count()} GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    writer = SummaryWriter(log_dir="experiments/logs/run_baseline")
    logger.info(f"Bắt đầu thực nghiệm: {config['experiment_name']}")
    logger.info(f"Cấu hình: {config}")
    
    for epoch in range(num_epochs):
        # ================= HUẤN LUYỆN (TRAIN) =================
        encoder.eval() # Pre-trained Encoder luôn đóng băng
        decoder.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for imgs, captions in train_loop:
            imgs, captions = imgs.to(device), captions.to(device)

            # Trích xuất đặc trưng (Không tính gradient cho ảnh)
            with torch.no_grad():
                features = encoder(imgs)
            
            # Forward pass 
            outputs = decoder(features, captions)
            
            # Reshape để tính loss
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = captions.reshape(-1)
            loss = criterion(outputs, targets) # Tính loss 

            # Backward pass và Cập nhật trọng số 
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradient tránh nổ đạo hàm cho LSTM
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

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
                
                features = encoder(imgs)
                outputs = decoder(features, captions)

                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = captions.reshape(-1)
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

    writer.close()
    logger.info("Huấn luyện hoàn tất!")