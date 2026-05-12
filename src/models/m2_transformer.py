import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Cơ chế Memory-Augmented Attention cho Encoder
# ---------------------------------------------------------
class MemoryAugmentedAttention(nn.Module):
    def __init__(self, embed_size, num_heads, memory_size=40):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        
        # "Learnable Memory" - Điểm mấu chốt của paper
        # Đây là một ma trận độc lập, model tự học ra các mẫu thị giác phổ biến
        self.memory_keys = nn.Parameter(torch.randn(1, memory_size, embed_size))
        self.memory_values = nn.Parameter(torch.randn(1, memory_size, embed_size))

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Nhân bản memory cho vừa với batch_size
        m_k = self.memory_keys.expand(batch_size, -1, -1)
        m_v = self.memory_values.expand(batch_size, -1, -1)
        
        # Nối (Concatenate) Memory vào chung với đặc trưng của ảnh
        extended_key = torch.cat([key, m_k], dim=1)
        extended_value = torch.cat([value, m_v], dim=1)
        
        # Tính Attention trên cả Ảnh + Bộ nhớ
        attn_output, _ = self.multihead_attn(query, extended_key, extended_value)
        return attn_output

class M2EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim=1024):
        super().__init__()
        self.memory_attn = MemoryAugmentedAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Self-attention với Memory
        attn_out = self.memory_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# ---------------------------------------------------------
# 2. Cơ chế Meshed Cross-Attention cho Decoder
# ---------------------------------------------------------
class MeshedDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, num_encoder_layers=3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        
        # Cross-attention cho từng lớp Encoder
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_size, num_heads, batch_first=True) 
            for _ in range(num_encoder_layers)
        ])
        
        # Cổng Gating để quyết định lấy bao nhiêu % thông tin từ lớp Encoder nào
        self.alpha_gates = nn.Linear(embed_size * num_encoder_layers, num_encoder_layers)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, x, encoder_outputs, tgt_mask):
        """
        encoder_outputs: List chứa đầu ra của TẤT CẢ các lớp Encoder
        """
        # 1. Masked Self-Attention (Không nhìn từ tương lai)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + attn_out)
        
        # 2. Meshed Cross-Attention
        cross_outs = []
        for i, enc_out in enumerate(encoder_outputs):
            c_out, _ = self.cross_attns[i](x, enc_out, enc_out)
            cross_outs.append(c_out)
            
        # Nối đầu ra của tất cả cross-attention lại
        concat_cross = torch.cat(cross_outs, dim=-1) # (batch, seq, embed * num_layers)
        
        # Tính trọng số Alpha (Hàm Sigmoid/Softmax)
        alphas = torch.softmax(self.alpha_gates(concat_cross), dim=-1)
        
        # Nhân trọng số với từng output và cộng dồn lại (Weighted Sum)
        meshed_out = sum(alphas[:, :, i:i+1] * cross_outs[i] for i in range(len(cross_outs)))
        
        x = self.norm2(x + meshed_out)
        
        # 3. Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x
    


class M2TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_seq_len=50):
        super().__init__()
        
        # 1. Lớp nhúng từ vựng và vị trí
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Sử dụng Learnable Positional Encoding cho đơn giản và hiệu quả
        self.pos_embed = nn.Embedding(max_seq_len, embed_size)
        self.dropout = nn.Dropout(0.1)

        # 2. M2 Encoders (Xử lý chuỗi đặc trưng ảnh)
        self.encoders = nn.ModuleList([
            M2EncoderLayer(embed_size, num_heads) for _ in range(num_layers)
        ])

        # 3. M2 Decoders (Xử lý Text + Nối lưới với toàn bộ Encoder)
        self.decoders = nn.ModuleList([
            MeshedDecoderLayer(embed_size, num_heads, num_encoder_layers=num_layers)
            for _ in range(num_layers)
        ])

        # 4. Lớp Linear cuối cùng để dự đoán xác suất từ vựng
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def generate_mask(self, size):
        """Tạo mặt nạ tam giác để model không 'nhìn trộm' từ tương lai khi huấn luyện"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, features, captions):
        # features: (batch_size, 49, embed_size)  - 49 là lưới ảnh 7x7
        # captions: (batch_size, seq_len)
        
        # Bỏ đi token <eos> cuối cùng ở input
        captions = captions[:, :-1]
        
        batch_size, seq_len = captions.size()
        device = features.device

        # --- PHẦN 1: XỬ LÝ ẢNH QUA BỘ NHỚ ---
        encoder_outputs = []
        x = features
        for enc in self.encoders:
            x = enc(x)
            encoder_outputs.append(x) # Lưu lại ĐẦU RA CỦA TỪNG LỚP để làm Meshed Attention

        # --- PHẦN 2: XỬ LÝ VĂN BẢN ---
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(device)
        word_embed = self.embed(captions)
        pos_embed = self.pos_embed(positions)
        tgt = self.dropout(word_embed + pos_embed)

        # Tạo mặt nạ che giấu (Target Mask)
        tgt_mask = self.generate_mask(seq_len).to(device)

        # --- PHẦN 3: ĐƯA VÀO MESHED DECODER ---
        out = tgt
        for dec in self.decoders:
            # Mỗi lớp Decoder đều được nhìn thấy TẤT CẢ đầu ra của Encoder
            out = dec(out, encoder_outputs, tgt_mask)

        # --- PHẦN 4: DỰ ĐOÁN TỪ TIẾP THEO ---
        preds = self.fc_out(out) # Output: (batch_size, seq_len, vocab_size)
        return preds

    def sample(self, features, vocab, max_len=20):
        """
        Hàm sinh từ autoregressive dành riêng cho Transformer.
        """
        device = features.device
        
        # 1. Khởi tạo mảng chứa câu với token <sos> đầu tiên
        sos_token = vocab.stoi['<sos>'] # Nếu dùng vocab tự viết, có thể là vocab.word2idx['<sos>']
        eos_token = vocab.stoi['<eos>']
        
        sampled_ids = [sos_token]
        
        for i in range(max_len):
            # 2. Chuyển mảng hiện tại thành Tensor (Shape: 1, current_seq_len)
            tgt = torch.tensor([sampled_ids], dtype=torch.long).to(device)
            
            # 3. Nhúng từ vựng và cộng vị trí
            tgt_embed = self.embed(tgt) # Hoặc self.embedding(tgt)
            if hasattr(self, 'pos_encoder'): 
                tgt_embed = self.pos_encoder(tgt_embed)
                
            # 4. Tạo mặt nạ che tương lai (Subsequent Mask)
            sz = tgt.size(1)
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
            
            # 5. Đưa qua Transformer Decoder
            # features đóng vai trò là 'memory' từ Encoder
            output = self.decoder(tgt=tgt_embed, memory=features, tgt_mask=tgt_mask)
            
            # 6. Dự đoán từ TIẾP THEO (Chỉ lấy kết quả ở vị trí token cuối cùng)
            preds = self.linear(output) # Hoặc self.fc_out(output)
            next_word_logits = preds[:, -1, :] # Shape: (1, vocab_size)
            
            _, predicted_id = next_word_logits.max(1)
            predicted_id = predicted_id.item()
            
            # 7. Nếu gặp <eos> thì ngắt mạch ngay tại Decoder
            if predicted_id == eos_token:
                break
                
            sampled_ids.append(predicted_id)
            
        # Trả về kết quả (Cắt bỏ token <sos> ở đầu mảng để câu in ra được tự nhiên)
        return sampled_ids[1:]
