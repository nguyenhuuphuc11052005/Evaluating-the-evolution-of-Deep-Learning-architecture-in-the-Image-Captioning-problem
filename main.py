import os
import json
import torch
import argparse
import torchvision.transforms as transforms
# import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

from src.data.build_vocab import Vocabulary
from src.data.dataset import get_loader
from src.models.encoder import ResNet50Encoder,ResNet50SpatialEncoder
from src.models.decoder_lstm import LSTMDecoder
from src.models.m2_transformer import M2TransformerDecoder
from src.training.train import train_model
from src.utils import set_seed, load_config


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
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    
    # 3. Chuẩn bị Dữ liệu
    vocab = build_vocab_from_json(train_ann_file, freq_threshold=5)
    vocab_size = len(vocab)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = get_loader(root_dir, train_ann_file, vocab, transform, batch_size=batch_size,num_workers =2)
    val_loader = get_loader(root_dir, val_ann_file, vocab, transform, batch_size=batch_size, num_workers =2)
    
    # 4. Khởi tạo Mô hình dựa trên biến cấu hình
    embed_size = config['model']['embed_size']
    encoder = ResNet50Encoder(embed_size)
    
    if config['model']['type'] == 'lstm':
        print("-> Khởi tạo LSTM Decoder...")
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