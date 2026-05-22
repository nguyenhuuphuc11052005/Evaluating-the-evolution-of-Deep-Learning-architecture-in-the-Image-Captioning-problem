import torch
import torchvision.transforms as transforms

def generate_caption(image_tensor, encoder, decoder, vocab, device, model_type='lstm', mode='greedy', beam_width=5):
    """
    Hàm sinh caption hỗ trợ nhiều kiến trúc mô hình.
    - model_type: 'lstm', 'm2_transformer', hoặc 'vit_transformer'
    """
    # 1. Chuyển mô hình sang chế độ đánh giá
    if encoder is not None:
        # tránh lỗi mô hình VitTransformer vì không có encoder
        encoder.eval()
    decoder.eval()

    image_tensor = image_tensor.to(device)
    
    # Lấy index của các token đặc biệt (Dành cho dòng họ Transformer)
    start_idx = vocab.word2idx.get(vocab.start_word, 1)
    end_idx = vocab.word2idx.get(vocab.end_word, 2)

    with torch.no_grad():
        # --- LUỒNG 1: BASELINE LSTM ---
        if model_type == 'lstm':
            features = encoder(image_tensor)
            features = features.unsqueeze(1) # Thêm chiều sequence_length
            
            if mode == 'greedy':
                sampled_ids = decoder.sample(features)
            else:
                sampled_ids = decoder.sample_beam_search(features, beam_width=beam_width)[0]

        # --- LUỒNG 2: M2 TRANSFORMER ---
        elif model_type == 'm2_transformer':
            features = encoder(image_tensor) # Trích xuất lưới đặc trưng (VD: 49 x embed_size)
            
            if mode == 'greedy':
                sampled_ids = decoder.sample(features, start_idx, end_idx)
            else:
                sampled_ids = decoder.sample_beam_search(features, start_idx, end_idx, beam_width=beam_width)

        # --- LUỒNG 3: ViT + TRANSFORMER ---
        elif model_type == 'vit_transformer':
            # Với ViT, biến `decoder` chính là full_model (ôm trọn cả encoder)
            if mode == 'greedy':
                sampled_ids = decoder.sample(image_tensor, start_idx, end_idx)
            else:
                sampled_ids = decoder.sample_beam_search(image_tensor, start_idx, end_idx, beam_width=beam_width)
                
        else:
            raise ValueError("model_type không hợp lệ! Hãy chọn 'lstm', 'm2_transformer', hoặc 'vit_transformer'")

    # Chuyển đổi chuỗi ID thành từ vựng thực tế
    words = []
    for word_id in sampled_ids:
        if torch.is_tensor(word_id):
            word_id = word_id.item()

        word = vocab.idx2word[word_id]
        
        # Dừng lại nếu gặp token kết thúc câu
        if word == vocab.end_word:
            break
            
        # Bỏ qua các token đặc biệt
        if word not in [vocab.start_word, vocab.pad_word, vocab.unk_word]:
            words.append(word)

    caption = " ".join(words)
    return caption