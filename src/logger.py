import logging
import os
from datetime import datetime

def setup_logger(experiment_name):
    """
    Thiết lập logger để ghi cả ra console và file .log
    """
    # Tạo thư mục chứa log nếu chưa có
    log_dir = os.path.join("experiments", "logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Tên file log dựa trên thời gian chạy
    log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Cấu hình logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Định dạng: Thời gian - Cấp độ - Thông báo
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Ghi ra File
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. Ghi ra Console (Màn hình)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_dir