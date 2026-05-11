import math
import torch
import torch.nn as nn
import torchvision.models as models

# 1. BỘ MÃ HÓA VỊ TRÍ (Dành cho Text)
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Cộng Positional Encoding vào vector nhúng từ vựng
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2. VISION TRANSFORMER ENCODER
class ViTEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ViTEncoder, self).__init__()
        # Tải ViT-Base với kích thước patch 16x16
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # Chúng ta cần chuỗi các patches, KHÔNG PHẢI kết quả phân loại cuối cùng
        self.patch_embed = vit.conv_proj
        self.class_token = vit.class_token
        self.seq_length = vit.seq_length
        self.pos_embedding = vit.encoder.pos_embedding
        self.encoder_layers = vit.encoder.layers
        self.ln = vit.encoder.ln
        
        # Đóng băng ViT để tránh nổ RAM
        for param in self.parameters():
            param.requires_grad = False
            
        # ViT-Base output vector 768 chiều. Cần một lớp Linear để ép về embed_size
        self.proj = nn.Linear(768, embed_size)

    def forward(self, x):
        # Trích xuất các mảnh ghép (patches) giống hệt mã nguồn của PyTorch
        n = x.shape[0]
        x = self.patch_embed(x)      # Đưa qua mạng chập để lấy patch
        x = x.flatten(2).transpose(1, 2)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        
        # Đưa qua các khối Transformer Encoder
        x = self.encoder_layers(x)
        x = self.ln(x)               # Shape: (batch_size, 197, 768)
        
        # Ép chiều không gian về embed_size của Decoder
        x = self.proj(x)             # Shape: (batch_size, 197, embed_size)
        return x

# 3. MÔ HÌNH TỔNG THỂ (ViT + Transformer Decoder)
class ViTCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_decoder_layers=3, dropout=0.1):
        super(ViTCaptioningModel, self).__init__()
        
        self.encoder = ViTEncoder(embed_size)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        
        # Lớp Transformer Decoder được tích hợp sẵn của PyTorch (Siêu tối ưu)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=embed_size * 4, 
            dropout=dropout,
            batch_first=True # Rất quan trọng: Báo cho mô hình biết Batch_size nằm ở chiều số 0
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # Che giấu từ tương lai, không cho model nhìn trộm khi học
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images, captions):
        # 1. Mã hóa ảnh thành chuỗi 197 vector đặc trưng
        memory = self.encoder(images) 
        
        # 2. Xử lý câu Text (Bỏ đi token <eos> ở cuối)
        captions = captions[:, :-1]
        tgt = self.embedding(captions)
        tgt = self.pos_encoder(tgt)
        
        # 3. Tạo mặt nạ tam giác
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(images.device)
        
        # 4. Đưa vào Transformer Decoder
        # tgt: Câu văn hiện tại (đã che tương lai)
        # memory: Chuỗi các patch ảnh từ ViT
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        
        # 5. Dự đoán từ tiếp theo
        preds = self.fc_out(out) # Shape: (batch_size, seq_len, vocab_size)
        
        return preds