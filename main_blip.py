import os
import argparse
import yaml
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator  # THÊM DÒNG NÀY

from src.data.blip_dataset import get_blip_loader
from src.training.train_blip import train_blip_model
from src.logger import setup_logger

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    exp_name = config['experiment_name']
    
    # KHỞI TẠO ACCELERATOR NGAY TỪ ĐẦU
    accelerator = Accelerator()
    
    logger, log_dir = setup_logger(exp_name)
    
    # Chỉ cho phép tiến trình chính (GPU 0) in log ra màn hình
    if accelerator.is_local_main_process:
        logger.info(f"========== BẮT ĐẦU FINE-TUNE BLIP: {exp_name} ==========")
    
    tb_log_dir = os.path.join("experiments/logs", exp_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    checkpoint_dir = os.path.join("experiments/checkpoints", exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if accelerator.is_local_main_process:
        logger.info("-> Đang tải Processor và Model từ Hugging Face...")
    
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    # Đóng băng Vision Transformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
        
    # XÓA TOÀN BỘ ĐOẠN DATA PARALLEL VÀ THIẾT LẬP DEVICE CŨ Ở ĐÂY
    # KHÔNG CẦN .to(device) NỮA
    
    if accelerator.is_local_main_process:
        logger.info("-> Đang chuẩn bị DataLoader...")
        
    train_loader = get_blip_loader(
        config['data']['root_dir'], config['data']['train_ann_file'], 
        processor, batch_size=config['training']['batch_size'], is_train=True
    )
    val_loader = get_blip_loader(
        config['data']['root_dir'], config['data']['val_ann_file'], 
        processor, batch_size=config['training']['batch_size'], is_train=False
    )
    
    if accelerator.is_local_main_process:
        logger.info("-> Bắt đầu vòng lặp huấn luyện...")
        
    # Truyền thêm accelerator vào hàm train
    train_blip_model(train_loader, val_loader, model, processor, config, logger, writer, checkpoint_dir, accelerator)
    
    if accelerator.is_local_main_process:
        logger.info("========== HOÀN TẤT ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP model")
    parser.add_argument('--config', type=str, default='configs/blip_config.yaml')
    args = parser.parse_args()
    main(args.config)