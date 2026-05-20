import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def generate_caption(image_path, encoder, decoder, vocab, device, model_type='lstm', mode='greedy', beam_width=5):
    """
    Hàm sinh caption hỗ trợ nhiều kiến trúc mô hình.
    - model_type: 'lstm', 'm2_transformer', hoặc 'vit_transformer'
    """
    # 1. Chuyển mô hình sang chế độ đánh giá
    encoder.eval()
    decoder.eval()

    # 2. Chuẩn bị Transform chuẩn ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load và xử lý ảnh
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

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

    # 4. Chuyển đổi chuỗi ID thành từ vựng thực tế
    words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        
        # Dừng lại nếu gặp token kết thúc câu
        if word == vocab.end_word:
            break
            
        # Bỏ qua các token đặc biệt
        if word not in [vocab.start_word, vocab.pad_word, vocab.unk_word]:
            words.append(word)

    caption = " ".join(words)
    return caption, image


def show_inference(image_path, encoder, decoder, vocab, device, model_type='lstm'):
    """
    Hàm hiển thị ảnh và so sánh kết quả giữa Greedy và Beam Search.
    """
    print(f"Đang sinh câu (Architecture: {model_type})...")
    
    # Sinh câu bằng 2 phương pháp
    greedy_cap, img = generate_caption(image_path, encoder, decoder, vocab, device, model_type=model_type, mode='greedy')
    beam_cap, _ = generate_caption(image_path, encoder, decoder, vocab, device, model_type=model_type, mode='beam_search', beam_width=5)
    
    # Hiển thị kết quả
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    
    # Format tiêu đề cho dễ nhìn
    title_text = f"[{model_type.upper()}]\nGreedy: {greedy_cap}\nBeam Search: {beam_cap}"
    plt.title(title_text, fontsize=12, loc='left', pad=10)
    plt.tight_layout()
    plt.show()