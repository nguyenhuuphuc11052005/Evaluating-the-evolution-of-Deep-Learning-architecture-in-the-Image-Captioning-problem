import os
from tqdm import tqdm
import torch
import pickle
import argparse
import warnings
import json 
warnings.filterwarnings("ignore")
from src.data.dataset import get_eval_loader, get_transforms
from inference import generate_caption
from src.evaluation.compute_metric import get_eval_score, ComputeMetrics
from src.models.decoder_lstm import LSTMDecoder
from src.models.encoder import ResNet50Encoder, ResNet50SpatialEncoder
from src.models.m2_transformer import M2TransformerDecoder
from src.models.vit_transformer import ViTCaptioningModel
from src.utils import load_config, set_seed
from src.logger import setup_logger

def evaluate_model(args, encoder, decoder, vocab, device, logger=None, log_dir=None):
    '''
    Đánh giá chung cho toàn model
    '''
    # loader
    loader = get_eval_loader(args.images_path, args.ann_file, vocab, get_transforms())
    # logger
    prediction_logs = []

    references = []
    hypotheses = []
    metric_evaluator = ComputeMetrics() 
    with torch.no_grad():
        for idx, (image, caps) in enumerate(tqdm(loader, desc = f'EVALUATING {args.model_type}')):
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

            # tính toán cho từng ảnh 
            per_image_metrics = metric_evaluator.compute_single_image(references=refs, hypothesis=prediction)

            # lấy thông tin của ảnh để vẽ ảnh 
            image_id = loader.dataset.ids[idx]
            img_info = loader.dataset.coco.loadImgs(image_id)[0]
            file_name = img_info['file_name']
            image_path = os.path.join(args.images_path, file_name)

            # thêm log cho predictions
            prediction_logs.append({
                "index": idx,                                   # index: bắt đầu từ 0
                "image_id": image_id,                           # id của ảnh
                "file_name": file_name,                         # tên ảnh
                "image_path": image_path,                       # đường dẫn của ảnh 
                "prediction": prediction,                       # prediction: câu dự đoán từ mô hình
                "references": refs,                             # references: câu tham chiếu của dữ liệu 
                "BLEU-1": per_image_metrics['BLEU']["BLEU-1"],  # chỉ số bleu1
                "BLEU-2": per_image_metrics['BLEU']["BLEU-2"],  # bleu2
                "BLEU-3": per_image_metrics['BLEU']["BLEU-3"],  # bleu3
                "BLEU-4": per_image_metrics['BLEU']["BLEU-4"],  # bleu4
                "METEOR": per_image_metrics["METEOR"],          # meteor
                "ROUGE_L": per_image_metrics["ROUGE_L"],        # rouge-l
                "avg_score": per_image_metrics["AVERAGE"]       # average score
            })
            # số câu đã dự đoán 
            if logger and len(hypotheses) % 1000 == 0:
                logger.info(f"Evaluated {len(hypotheses)} samples")
        
    metrics = get_eval_score(references=references, hypotheses=hypotheses)
    # thêm log cho kết quả của metric
    if logger: 
        logger.info("Evaluate finished")
        logger.info(f"Metrics: {metrics}")
    # lưu predictions vào file json 
    if log_dir:
        version = 1

        while True:
            pred_path = os.path.join(log_dir,f"eval_predictions_ver{version}.json")

            if not os.path.exists(pred_path):
                break

            version += 1

        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(prediction_logs, f, ensure_ascii=False, indent=2)

        if logger:
            logger.info(f"Saved prediction logs to {pred_path}")

    return metrics

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
        encoder = ResNet50SpatialEncoder(embed_size).to(device)
        decoder = M2TransformerDecoder(vocab_size=vocab_size,embed_size=embed_size,num_heads=num_heads,
                                       num_layers=num_layers, max_seq_len=max_seq_len).to(device)
        encoder.load_state_dict(checkpoint.get('encoder', checkpoint.get('encoder_state_dict')))
        decoder.load_state_dict(checkpoint.get('decoder', checkpoint.get('decoder_state_dict')))

    elif args.model_type == 'vit_transformer':
        encoder = None
        embed_size = config['model']['embed_size']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        decoder = ViTCaptioningModel(vocab_size=vocab_size, embed_size=embed_size,
                                     num_heads=num_heads, num_decoder_layers=num_layers).to(device)
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

    # thêm đường dẫn cho logger 
    experiment_name = f"eval_{args.model_type}_{args.decode_mode}_{args.beam_size}"
    logger, log_dir = setup_logger(experiment_name=experiment_name)
    logger.info("Start evaluation")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Decode mode: {args.decode_mode}")
    logger.info(f"Beam size: {args.beam_size}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Images path: {args.images_path}")
    logger.info(f"Annotation file: {args.ann_file}")

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

    metrics = evaluate_model(args, encoder, decoder, vocab, device,
                             logger=logger, log_dir=log_dir)

    print(f"{args.decode_mode} | beam={args.beam_size}")
    print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
        (metrics['BLEU']["BLEU-1"],  metrics['BLEU']["BLEU-2"],  metrics['BLEU']["BLEU-3"],  metrics['BLEU']["BLEU-4"],
        metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))
