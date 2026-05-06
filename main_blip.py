import argparse
import yaml
from transformers import BlipProcessor, BlipForConditionalGeneration

# Import từ các file ta vừa viết
from src.data.blip_dataset import get_blip_loader
from src.training.train_blip import train_blip_model

def main(config_path):
    # 1. Đọc file cấu hình
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    print(f"=== BẮT ĐẦU DỰ ÁN FINE-TUNE BLIP ({config['experiment_name']}) ===")
    
    # 2. Tải Processor và Model từ Hugging Face
    print("-> Đang tải trọng số BLIP từ Hugging Face (Có thể mất vài phút)...")
    model_name = "Salesforce/blip-image-captioning-base"
    
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    # --- BÍ QUYẾT SINH TỒN (ĐÓNG BĂNG ViT) ---
    print("-> Đang đóng băng (Freeze) Vision Transformer để tiết kiệm VRAM...")
    for param in model.vision_model.parameters():
        param.requires_grad = False
    # CHỈ học (Unfreeze) phần text_decoder
        
    # 3. Chuẩn bị DataLoader
    root_dir = config['data']['root_dir']
    train_ann_file = config['data']['train_ann_file']
    val_ann_file = config['data']['val_ann_file']
    batch_size = config['training']['batch_size']
    
    print("-> Đang chuẩn bị tập dữ liệu...")
    train_loader = get_blip_loader(root_dir, train_ann_file, processor, batch_size=batch_size, is_train=True)
    val_loader = get_blip_loader(root_dir, val_ann_file, processor, batch_size=batch_size, is_train=False)
    
    # 4. Huấn luyện
    print("-> Khởi động vòng lặp huấn luyện...")
    train_blip_model(train_loader, val_loader, model, processor, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP model")
    parser.add_argument('--config', type=str, default='configs/blip_config.yaml', help='Đường dẫn tới file config')
    args = parser.parse_args()
    main(args.config)