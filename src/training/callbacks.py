import os
import torch
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, save_path='experiments/checkpoints/best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
        # Đảm bảo thư mục lưu checkpoint luôn tồn tại
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def __call__(self, val_loss, model_state, optimizer_state, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_state, optimizer_state, epoch)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"=> EarlyStopping: Không cải thiện ({self.counter} / {self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_state, optimizer_state, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, encoder, decoder, optimizer, epoch):
        print(f"   [Checkpoint] Validation Loss giảm ({self.best_loss:.4f} --> {val_loss:.4f}). Đã lưu mô hình!")
        
        # Trích xuất state_dict an toàn, xử lý cả trường hợp dùng DataParallel (Multi-GPU)
        enc_state = encoder.module.state_dict() if isinstance(encoder, torch.nn.DataParallel) else encoder.state_dict()
        dec_state = decoder.module.state_dict() if isinstance(decoder, torch.nn.DataParallel) else decoder.state_dict()

        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': enc_state,
            'decoder_state_dict': dec_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.save_path)


class ExperimentLogger:
    def __init__(self, log_dir="experiments/logs/run_01"):
        # Khởi tạo TensorBoard Writer
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"=> TensorBoard đang lưu log tại: {log_dir}")

    def log_metrics(self, train_loss, val_loss, epoch):
        # Ghi metric vào TensorBoard sau mỗi Epoch
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
    def close(self):
        self.writer.close()