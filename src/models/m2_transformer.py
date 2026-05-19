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

    def sample(self, features, start_idx, end_idx, max_len=20):
        """
        Inference bằng Greedy Search dành riêng cho Meshed-Memory Transformer.
        - features: (1, 49, embed_size) - Lưới đặc trưng ảnh
        """
        device = features.device
        
        # --- PHẦN 1: XỬ LÝ ẢNH QUA M2 ENCODER (Chỉ chạy 1 lần) ---
        encoder_outputs = []
        x = features
        for enc in self.encoders:
            x = enc(x)
            encoder_outputs.append(x) # Lưu lại list các output để Meshed Decoder dùng
            
        # --- PHẦN 2: KHỞI TẠO CHUỖI TỪ VỰNG ---
        # Bắt đầu với token <start>
        tgt = torch.full((1, 1), start_idx, dtype=torch.long, device=device)
        
        # --- PHẦN 3: VÒNG LẶP AUTOREGRESSIVE ---
        for i in range(max_len):
            seq_len = tgt.size(1)
            
            # Xử lý Vị trí và Nhúng từ vựng hiện tại
            positions = torch.arange(0, seq_len).unsqueeze(0).to(device)
            word_embed = self.embed(tgt)
            pos_embed = self.pos_embed(positions)
            out = word_embed + pos_embed
            
            # Tạo mặt nạ tam giác
            tgt_mask = self.generate_mask(seq_len).to(device)
            
            # Đưa qua từng lớp Meshed Decoder
            for dec in self.decoders:
                # Đưa cả list encoder_outputs vào
                out = dec(out, encoder_outputs, tgt_mask)
                
            # Lấy vector của TỪ CUỐI CÙNG trong chuỗi
            preds = self.fc_out(out[:, -1, :]) # Shape: (1, vocab_size)
            
            # Chọn từ có xác suất cao nhất
            next_word = preds.argmax(dim=1, keepdim=True)
            
            # Nối từ mới vào chuỗi
            tgt = torch.cat([tgt, next_word], dim=1)
            
            # Dừng nếu gặp token <end>
            if next_word.item() == end_idx:
                break
                
        # Trả về list ID (cắt bỏ token <start> ở vị trí số 0)
        return tgt[0, 1:].tolist()

    def sample_beam_search(self, features, start_idx, end_idx, max_len=20, beam_width=5):
        """
        Inference bằng Beam Search dành riêng cho Meshed-Memory Transformer.
        """
        device = features.device
        
        # --- PHẦN 1: XỬ LÝ ẢNH QUA M2 ENCODER (Chỉ chạy 1 lần) ---
        encoder_outputs = []
        x = features
        for enc in self.encoders:
            x = enc(x)
            encoder_outputs.append(x)
            
        # --- PHẦN 2: KHỞI TẠO BEAM ---
        # Lưu các nhánh tiềm năng: list các tuple (chuỗi_ids, tổng_log_prob)
        k_beams = [([start_idx], 0.0)]
        
        # --- PHẦN 3: VÒNG LẶP TÌM KIẾM ---
        for step in range(max_len):
            new_beams = []
            
            for seq, score in k_beams:
                # Nếu nhánh này đã kết thúc, giữ nguyên nó lại
                if seq[-1] == end_idx:
                    new_beams.append((seq, score))
                    continue
                    
                # Chuyển chuỗi thành Tensor
                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                seq_len = tgt.size(1)
                
                # Embedding + Position
                positions = torch.arange(0, seq_len).unsqueeze(0).to(device)
                word_embed = self.embed(tgt)
                pos_embed = self.pos_embed(positions)
                out = word_embed + pos_embed
                
                # Tạo Mask
                tgt_mask = self.generate_mask(seq_len).to(device)
                
                # Chạy qua Decoder
                for dec in self.decoders:
                    out = dec(out, encoder_outputs, tgt_mask)
                    
                # Dự đoán từ cuối cùng
                preds = self.fc_out(out[:, -1, :])
                
                # Tính Log Softmax (Cực kỳ quan trọng cho Beam Search)
                log_probs = torch.nn.functional.log_softmax(preds, dim=-1).squeeze(0)
                
                # Lấy Top K từ tốt nhất cho nhánh này
                topk_log_probs, topk_idx = log_probs.topk(beam_width)
                
                # Tạo ra K nhánh mới từ nhánh hiện tại
                for i in range(beam_width):
                    next_word = topk_idx[i].item()
                    next_score = score + topk_log_probs[i].item()
                    new_beams.append((seq + [next_word], next_score))
                    
            # Sắp xếp tất cả các nhánh mới sinh ra theo điểm số giảm dần
            k_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Nếu tất cả các nhánh tốt nhất đều đã chạm <end>, ta dừng vòng lặp sớm
            if all(seq[-1] == end_idx for seq, _ in k_beams):
                break
                
        # Lấy chuỗi của nhánh đứng đầu bảng
        best_seq = k_beams[0][0]
        
        # Trả về list ID (cắt bỏ token <start>)
        return best_seq[1:]
