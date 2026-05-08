import os
import json 
import torch
import warnings
# ghi log
# from logger import logging 
from typing import List
from src.evaluation.compute_metric import ComputeMetrics
from tqdm import tqdm

class Evaluator:
    '''
    Lớp tính các chỉ số trên tập test 
    '''
    def __init__(self, encoder, search_method, vocab, test_loader,
                checkpoint_path, device='cuda'):
        
        '''
        Khởi tạo Evaluator:
        :param encoder: mô hình encoder
        :param search_method: thuật toán tìm kiếm sinh caption
        :param vocab: vocabulary object
        :param test_loader: DataLoader của tập test
        :param checkpoint_path: đường dẫn checkpoint tốt nhất
        :param device: cpu hoặc cuda
        '''
        self.encoder = encoder.to(device)

        self.search = search_method # chứa decoder bên trong 
        self.vocab = vocab
        self.test_loader = test_loader
        
        self.device = device        
        self.metric_computer = ComputeMetrics()
        self.load_checkpoint(checkpoint_path)

    # load checkpoint
    def load_checkpoint(self, checkpoint_path):
        '''
        Load những tọng số tốt nhất của mô hình để sinh caption trên tập test
        :param checkpoint_path: đường dẫn đến file checkpoint (.pth)
        '''
        
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )
        # load trọng số của encoder
        self.encoder.load_state_dict(
            checkpoint['encoder_state_dict']
        )
        # load trọng số của decoder
        self.search.decoder.load_state_dict(
            checkpoint['decoder_state_dict']
        )
        # chuyển model sang chế độ evaluation 
        self.encoder.eval()
        self.search.decoder.eval()

        print(f"Checkpoint loaded successfully!")

    def decode_caption(self, token_ids: List[int]) -> str:
        '''
        Chuyển danh sách token ids thành caption dạng chuỗi.
        :param token_ids: danh sách các token ids 
        :return: caption tương ứng (str)
        '''
        words = []
        for idx in token_ids:
            word = self.vocab.itos[idx]
            # kết thúc câu 
            if word == "<eos>":
                break
            # bỏ token đặc biệt
            if word not in ["<pad>", "<sos>"]:
                words.append(word)

        return " ".join(words)                

    def generate_caption(self, features) -> str:
        '''
        Sinh caption từ feature vector ảnh
        :param features: vector đặc trưng ảnh 
        :return: caption dự đoán
        '''
        # sinh token ids
        token_ids = self.search.generate(features)
        # sinh caption 
        caption = self.decode_caption(token_ids=token_ids)
        return caption
     
    def generate_predictions(self)->List[str]:
        '''
        Sinh caption cho toàn bộ tập test
        '''
        predictions = []
        with torch.no_grad():
            for batch, _, _ in tqdm(self.test_loader, desc='Generating Captions'):
                images = batch[0].to(self.device)
                # encode image
                features = self.encoder(images)
                # generate caption 
                caption = self.generate_caption(features=features)
                # thêm vào danh sách
                predictions.append(caption)
        return predictions

    def evaluate(self):
        '''
        Đánh giá mô hình bằng các chỉ số: BLEU, ROUGE, METOR, CIDEr
        '''
        # Danh sách lưu tham chiếu (true captions) và giả thuyết (prediction) cho từng tấm hình
        # Nếu cho n tấm ảnh, ta có n giả thuyết và tham chiếu a, b, c,... cho từng tấm ảnh, ta cần
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for image, _, _, allcaps in tqdm(self.test_loader, desc='Evaluating'):
                image = image.to(self.device)

                # encode
                features = self.encoder(image)

                # sinh prediction 
                token_ids = self.search.generate(features)

                # sinh caption dự đoán
                pred_caption = [
                    self.vocab.itos[idx]
                    for idx in token_ids
                    if self.vocab.itos[idx]
                    not in ['<sos>', '<eos>', '<pad>']
                ]

                hypotheses.append(pred_caption)
                
                # grouth truth caption 
                img_caps = allcaps[0].tolist()
                gt_captions = []

                for cap in img_caps:

                    tokens = [
                        self.vocab.itos[idx]
                        for idx in cap
                        if self.vocab.itos[idx]
                        not in ['<sos>', '<eos>', '<pad>']
                    ]

                    gt_captions.append(tokens)

                references.append(gt_captions)
                # sanity check
                assert len(references) == len(hypotheses)

        # tính toán chỉ số
        metrics = self.metric_computer.compute_all(
            references,
            hypotheses
        )

        return metrics
    
    def save_predictions(self, save_path:str='prediction.json')->None:
        '''
        Lưu caption đã dự đoán ra file json
        '''
        results = []

        with torch.no_grad():

            for images, _, _, _ in tqdm(
                self.test_loader,
                desc='Saving Predictions'
            ):

                images = images.to(self.device)
                
                # encode
                features = self.encoder(images)
                
                # dự đoán caption
                caption = self.generate_caption(features)

                results.append({
                    'caption': caption
                })

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(f'Saved predictions to: {save_path}')