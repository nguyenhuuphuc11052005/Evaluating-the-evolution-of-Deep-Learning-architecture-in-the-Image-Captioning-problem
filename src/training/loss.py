import torch
import torch.nn as nn

class CaptionLoss(nn.Module):
    def __init__(self, pad_idx):
        super(CaptionLoss, self).__init__()
        # Bỏ qua token <pad> để không bị nhiễu loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, outputs, targets):
        """
        Tính Cross Entropy Loss cho tập chuỗi văn bản.
        - outputs: Tensor dự đoán từ mô hình (batch_size, seq_length, vocab_size)
        - targets: Tensor nhãn thực tế (batch_size, seq_length)
        """
        # Reshape lại để phù hợp với hàm CrossEntropyLoss của PyTorch
        outputs = outputs.reshape(-1, outputs.shape[2])
        targets = targets.reshape(-1)
        
        loss = self.criterion(outputs, targets)
        return loss
    


def get_criterion(vocab):
    pad_idx = vocab.stoi["<pad>"]
    # Trả về hàm CrossEntropyLoss chuẩn của PyTorch
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,label_smoothing=0.1) # Thêm label smoothing để cải thiện độ ổn định khi huấn luyện
    return criterion