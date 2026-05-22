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
from src.utils import load_config, set_seed

def evaluate_lstm(args, encoder, decoder, vocab, device):
    '''
    Đánh giá mô hình LSTM
    '''
    # loader
    loader = get_eval_loader(args.images_path, args.ann_file, vocab, get_transforms())

    references = []
    hypotheses = []

    with torch.no_grad():
        for image, caps in tqdm(loader, desc = 'EVALUATING WITH LSTM MODEL'):
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

def evaluate_m2(args, encoder, decoder, vocab, device):
    pass

def evaluate_vit(args, encoder, decoder, vocab, device):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image_Captioning_Evaluating')
    parser.add_argument("--config",default="configs/baseline_lstm.yaml")
    parser.add_argument("--checkpoint", default="/experiments/checkpoints/baseline_lstm_run1/best_model.pth")
    parser.add_argument("--model_type", default="lstm")
    parser.add_argument("--decode_mode",default="beam_search")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--images_path", default="/root/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014/images/train2014")
    parser.add_argument("--ann_file", default="/data/ms_coco/annotations/subset_test.json")
    args = parser.parse_args()

    # 1. Khởi tạo device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = 'vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"-> Đã load Vocab với {len(vocab)} từ.")
    vocab_size = len(vocab)
    # 2. Load config, khởi tạo lại cấu trúc mạng 
    config = load_config(args.config)
    set_seed()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    embed_size = config['model']['embed_size']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    encoder = ResNet50Encoder(embed_size).to(device)
    decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)

    encoder.load_state_dict(checkpoint.get('encoder', checkpoint.get('encoder_state_dict')))
    decoder.load_state_dict(checkpoint.get('decoder', checkpoint.get('decoder_state_dict')))

    metrics = evaluate_lstm(args, encoder, decoder, vocab, device)
    print(f"{args.decode_mode} | beam={args.beam_size}")
    print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
        (metrics['BLEU']["BLEU-1"],  metrics['BLEU']["BLEU-2"],  metrics['BLEU']["BLEU-3"],  metrics['BLEU']["BLEU-4"],
        metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))