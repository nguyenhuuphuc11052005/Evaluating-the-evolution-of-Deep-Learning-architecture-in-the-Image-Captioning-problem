import os
import json
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
import pickle
import warnings
warnings.filterwarnings("ignore")

from src.data.build_vocab import Vocabulary
from src.data.dataset import get_loader
from src.models.encoder import ResNet50Encoder,ResNet50SpatialEncoder
from src.models.decoder_lstm import LSTMDecoder
from src.models.m2_transformer import M2TransformerDecoder
from src.training.train import train_model
from src.utils import set_seed, load_config
from src.models.vit_transformer import ViTCaptioningModel

def build_vocab_from_json(json_path, freq_threshold=5):
    print(f"Đang phân tích ngôn ngữ từ: {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    captions = [ann['caption'] for ann in data['annotations']]
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions)
    return vocab

def main(config_path):
    # dist.init_process_group(backend="nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # 1. Đọc file cấu hình YAML
    config = load_config(config_path)
    print(f"=== Đang chạy thực nghiệm: {config['experiment_name']} ===")

    set_seed(42)

    # 2. Lấy tham số từ config
    root_dir = config['data']['root_dir']
    train_ann_file = config['data']['train_ann_file']
    val_ann_file = config['data']['val_ann_file']
    
    batch_size = config['training']['batch_size']
    # learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    
    # 3. Chuẩn bị Dữ liệu
    vocab = build_vocab_from_json(train_ann_file, freq_threshold=5)
    vocab_size = len(vocab)
    
    checkpoint_dir = os.path.join("experiments/checkpoints", config['experiment_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    vocab_save_path = os.path.join(checkpoint_dir, "vocab.pkl")
    
    # Chỉ cần lưu 1 lần (nếu chạy DataParallel/1 GPU)
    with open(vocab_save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"-> Đã đóng gói và lưu bộ từ điển (Vocab) tại: {vocab_save_path}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)), # Cắt ngẫu nhiên
        transforms.RandomHorizontalFlip(p=0.5), # Lật ngang ngẫu nhiên 50%
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Đổi màu nhẹ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = get_loader(root_dir, train_ann_file, vocab, train_transform, batch_size=batch_size,num_workers =2)
    val_loader = get_loader(root_dir, val_ann_file, vocab, val_transform, batch_size=batch_size, num_workers =2)
    
    # 4. Khởi tạo Mô hình dựa trên biến cấu hình
    embed_size = config['model']['embed_size']
    encoder = ResNet50Encoder(embed_size)
    
    if config['model']['type'] == 'lstm':
        print("-> Khởi tạo LSTM Decoder...")
        encoder_lr = config['training']['encoder_lr']
        decoder_lr = config['training']['decoder_lr']
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']
        decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)
    elif config['model']['type'] == 'm2_transformer':
        print("-> Khởi tạo SOTA: ResNet50 (Spatial) + M2 Transformer Decoder...")
        # Sử dụng Encoder lấy lưới 7x7 thay vì vector 1D
        encoder = ResNet50SpatialEncoder(embed_size) 
        
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        max_seq_len = config['model']['max_seq_len']
        decoder = M2TransformerDecoder(vocab_size, embed_size, num_heads, num_layers, max_seq_len)
    
    elif config['model']['type'] == 'vit_transformer':
        print("-> Khởi tạo mô hình: ViT Encoder + Transformer Decoder...")
        embed_size = config['model']['embed_size']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        
        # Model này đã bọc sẵn cả Encoder và Decoder bên trong  
        # Ta khởi tạo nó, sau đó tách nó ra thành biến encoder/decoder giả để đưa vào hàm train cũ
        full_model = ViTCaptioningModel(vocab_size, embed_size, num_heads, num_layers)
        
        # MẸO: Hàm train_model của bạn yêu cầu truyền vào `encoder` và `decoder` rời nhau.
        # Ở đây ta lừa hàm train một chút: `encoder` chỉ đóng vai trò truyền hình nộm,
        # vì `full_model` (được gán cho biến `decoder`) sẽ ôm đồm làm toàn bộ việc forward pass.
        encoder = nn.Identity() 
        decoder = full_model
    # 5. Huấn luyện 
    train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        encoder=encoder, 
        decoder=decoder, 
        vocab=vocab, 
        config=config  
    )

if __name__ == "__main__":
    # Sử dụng Argparse để truyền đường dẫn file config từ Terminal
    parser = argparse.ArgumentParser(description="Image Captioning Project")
    parser.add_argument('--config', type=str, default='configs/baseline_lstm.yaml', 
                        help='Đường dẫn tới file cấu hình YAML')
    args = parser.parse_args()
    
    main(args.config)
