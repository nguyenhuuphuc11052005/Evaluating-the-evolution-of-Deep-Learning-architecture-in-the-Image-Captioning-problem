import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        
        # Lớp Embedding: Chuyển index của từ thành vector ngữ nghĩa
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Mạng LSTM cốt lõi
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Lớp Linear cuối cùng để dự đoán từ vựng (ánh xạ về vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Dropout để chống Overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features shape: (batch_size, embed_size)
        # captions shape: (batch_size, max_seq_length)
        
        # BỎ ĐI TOKEN CUỐI CÙNG CỦA CAPTION
        # Vì nếu câu là <sos> con chó đang chạy <eos>
        # Input đưa vào sẽ là: <sos> con chó đang chạy (để dự đoán từ tiếp theo)
        embeddings = self.dropout(self.embed(captions[:, :-1])) # (batch_size, max_seq_length - 1, embed_size)
        
        # NỐI VECTOR ẢNH VÀO ĐẦU CHUỖI VĂN BẢN
        # Thêm 1 chiều time-step cho vector ảnh: (batch_size, 1, embed_size)
        features = features.unsqueeze(1)
        
        # Chuỗi input hoàn chỉnh cho LSTM: [Ảnh, Từ_1, Từ_2, ...]
        embeddings = torch.cat((features, embeddings), dim=1) 
        
        # Chạy qua mạng LSTM
        # hiddens shape: (batch_size, max_seq_length, hidden_size)
        hiddens, _ = self.lstm(embeddings)
        
        # Dự đoán xác suất cho từ tiếp theo
        outputs = self.linear(hiddens) # (batch_size, max_seq_length, vocab_size)
        
        return outputs
    
    def sample(self, features, max_len=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None # Trạng thái ẩn ban đầu
        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
             
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            
            inputs = self.embed(predicted).unsqueeze(1)
            
        return sampled_ids
