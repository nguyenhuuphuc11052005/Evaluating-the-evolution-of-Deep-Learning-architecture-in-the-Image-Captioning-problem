import os
from tqdm import tqdm
import torch
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")
from src.data.dataset import get_eval_loader, get_transforms
from inference import generate_caption
from src.evaluation.compute_metric import get_eval_score
from src.models.decoder_lstm import LSTMDecoder
from src.models.encoder import ResNet50Encoder
from src.models.m2_transformer import M2TransformerDecoder
from src.models.vit_transformer import ViTCaptioningModel
from src.utils import load_config, set_seed

def evaluate_model(args, encoder, decoder, vocab, device):
    '''
    Đánh giá chung cho toàn model
    '''
    # loader
    loader = get_eval_loader(args.images_path, args.ann_file, vocab, get_transforms())

    references = []
    hypotheses = []

    with torch.no_grad():
        for image, caps in tqdm(loader, desc = f'EVALUATING {args.model_type}'):
            image = image.to(device)

            prediction = generate_caption(image, encoder, decoder, vocab, device, 
                                          model_type=args.model_type, mode=args.decode_mode, beam_width=args.beam_size)

            hypotheses.append(prediction)

            refs = []
            # lấy tất cả caption của ảnh 
            for cap in caps[0]:
                ref = " ".join([
                    vocab.idx2word[idx.item()]
                    for idx in cap
                    if idx.item() not in {
                        vocab.word2idx["<pad>"],
                        vocab.word2idx["<start>"],
                        vocab.word2idx["<end>"]
                    }
                ])
                refs.append(ref)

            references.append(refs)
    return get_eval_score(references=references, hypotheses=hypotheses)

def build_model(args, config, vocab_size, device):
    '''
    khởi tạo model theo model_type
    ''' 
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.model_type == 'lstm':
        embed_size = config['model']['embed_size']
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']
        encoder = ResNet50Encoder(embed_size).to(device)
        decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
        encoder.load_state_dict(checkpoint.get('encoder', checkpoint.get('encoder_state_dict')))
        decoder.load_state_dict(checkpoint.get('decoder', checkpoint.get('decoder_state_dict')))

    elif args.model_type == 'm2_transformer':
        embed_size = config['model']['embed_size']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        max_seq_len = config['model']['max_seq_len']
        encoder = ResNet50Encoder(embed_size).to(device)
        decoder = M2TransformerDecoder(vocab_size=vocab_size,embed_size=embed_size,num_heads=num_heads,
                                       num_layers=num_layers, max_seq_len=max_seq_len).to(device)
        encoder.load_state_dict(checkpoint.get('encoder', checkpoint.get('encoder_state_dict')))
        decoder.load_state_dict(checkpoint.get('decoder', checkpoint.get('decoder_state_dict')))

    elif args.model_type == 'vit_transformer':
        encoder = None
        embed_size = config['model']['embed_size']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        decoder = ViTCaptioningModel(vocab_size=vocab_size, embed_size=embed_size,num_heads=num_heads).to(device)
        decoder.load_state_dict(checkpoint.get('decoder', checkpoint.get('decoder_state_dict')))
    else:
        raise ValueError("Sai model_type!")
    
    return encoder, decoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image_Captioning_Evaluating')
    parser.add_argument("--config",required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=['lstm', 'm2_transformer', 'vit_transformer'])
    parser.add_argument("--decode_mode",default="beam_search")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--images_path", required=True)
    parser.add_argument("--ann_file", required=True)
    args = parser.parse_args()
    set_seed()
    # 1. Khởi tạo device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = 'vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"-> Đã load Vocab với {len(vocab)} từ.")
    vocab_size = len(vocab)
    # 2. Load config, khởi tạo lại cấu trúc mạng 
    config = load_config(args.config)
    encoder, decoder = build_model(args, config=config, vocab_size=vocab_size, device=device)

    metrics = evaluate_model(args, encoder, decoder, vocab, device)

    print(f"{args.decode_mode} | beam={args.beam_size}")
    print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
        (metrics['BLEU']["BLEU-1"],  metrics['BLEU']["BLEU-2"],  metrics['BLEU']["BLEU-3"],  metrics['BLEU']["BLEU-4"],
        metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))