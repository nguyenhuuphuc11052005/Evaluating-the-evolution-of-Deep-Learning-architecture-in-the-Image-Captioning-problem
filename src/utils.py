import torch
import numpy as np
import random
import os
import yaml

def set_seed(seed=42):
    """
    Cố định tất cả các seed ngẫu nhiên để đảm bảo kết quả có thể tái lập.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # Cố định cho PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Bắt buộc nếu dùng nhiều GPU
    
    # Đảm bảo các thuật toán cuộn (convolution) của cuDNN hoạt động nhất quán
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[*] Đã cố định Random Seed: {seed}")



def load_config(config_path):
    """
    Hàm hỗ trợ đọc file YAML và trả về một dictionary chứa cấu hình.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[*] Đã nạp cấu hình từ {config_path}")
    return config