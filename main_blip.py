import os
import argparse
import yaml
import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter

# Import các module của dự án
from src.data.blip_dataset import get_blip_loader
from src.training.train_blip import train_blip_model
from src.logger import setup_logger

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    exp_name = config['experiment_name']
    
    # 1. KHỞI TẠO LOGGER & THƯ MỤC
    logger, log_dir = setup_logger(exp_name)
    logger.info(f"========== BẮT ĐẦU FINE-TUNE BLIP: {exp_name} ==========")
    
    tb_log_dir = os.path.join("experiments/logs", exp_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    checkpoint_dir = os.path.join("experiments/checkpoints", exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2. TẢI MÔ HÌNH HUGGING FACE
    logger.info("-> Đang tải Processor và Model từ Hugging Face...")
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    # 3. ĐÓNG BĂNG VISION TRANSFORMER (CHỈ TRAIN TEXT DECODER)
    logger.info("-> Đang đóng băng Vision Transformer để tiết kiệm VRAM...")
    for param in model.vision_model.parameters():
        param.requires_grad = False
        
    # 4. THIẾT LẬP THIẾT BỊ & DATAPARALLEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logger.info(f"-> Phát hiện {torch.cuda.device_count()} GPUs. Đang kích hoạt nn.DataParallel!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # 5. CHUẨN BỊ DATA LOADER
    logger.info("-> Đang chuẩn bị DataLoader...")
    train_loader = get_blip_loader(
        config['data']['root_dir'], 
        config['data']['train_ann_file'], 
        processor, 
        batch_size=config['training']['batch_size'], 
        is_train=True
    )
    val_loader = get_blip_loader(
        config['data']['root_dir'], 
        config['data']['val_ann_file'], 
        processor, 
        batch_size=config['training']['batch_size'], 
        is_train=False
    )
    
    # 6. BẮT ĐẦU HUẤN LUYỆN
    logger.info("-> Bắt đầu vòng lặp huấn luyện...")
    train_blip_model(train_loader, val_loader, model, processor, config, logger, writer, checkpoint_dir)
    logger.info("========== HOÀN TẤT ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP model")
    parser.add_argument('--config', type=str, default='configs/blip_config.yaml', help='Đường dẫn tới file config')
    args = parser.parse_args()
    main(args.config)