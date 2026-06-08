import torch
import torchvision.transforms as transforms
import json
import os
import textwrap
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from src.evaluation.eval import build_model
import argparse
import pickle
from utils import load_config

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

def wrap_multiline_text(text: str, width:int=90) -> str:
    '''
    Tự động xuống dòng cho một chuỗi văn bản dài
    :param text: chuỗi văn bản cần xuống dòng 
    :param width: số ký tự tối đa trên 1 dòng 
    :return: chuỗi văn bản đã được xuống dòng 
    '''
    # tách văn bản có sẵn
    lines = text.split("\n")
    # danh sách lưu các dòng sau khi wrap
    wrapped_lines = []

    for line in lines:
        # nếu dòng rỗng thì giữ nguyên
        if line.strip() == "":
            wrapped_lines.append("")
        else:
            # nếu dòng có nội dung thì wrap theo độ rộng
            wrapped_lines.extend(textwrap.wrap(line, width=width))
    # ghép các dòng thành chuỗi hoàn chỉnh 
    return "\n".join(wrapped_lines)

def load_json(json_path: str) -> List[Dict[str, Any]]:
    '''
    Đọc file JSON chứa kết quả đánh giá caption của mô hình.
    :param json_path: Đường dẫn tới file JSON
    :return: Danh sách các dictionary, mỗi dictionary tương ứng với một ảnh
    '''

    # Kiểm tra file JSON có tồn tại hay không
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file JSON: {json_path}")

    # Mở và đọc file JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Kiểm tra dữ liệu sau khi đọc có đúng dạng list không
    if not isinstance(data, list):
        raise ValueError("File JSON phải có dạng List[Dict], tức là danh sách các item kết quả.")

    return data

def select_lowest_score_items(data: List[Dict[str, Any]], score_key: str = 'avg_score',
                              top_k: int = 10) -> List[Dict[str, Any]]:
    '''
    Chọn top_k ảnh có điểm thấp nhất theo metric cụ thể
    :param data: danh sách kết quả đọc từ file json 
    :param score_key: tên metric để sắp xếp, mặc định lấy trung bình (avg_score)
    :param top_k: số lượng ảnh cần lấy ra, mặc định là 10
    :return: danh sách top_k items có score thấp nhất
    '''
    # lọc ra những item có tồn tại score_key và giá trị không phải None
    valid_items = [
        item for item in data
        if score_key in item and item[score_key] is not None
    ]

    # nếu không có item hợp lệ thì báo lỗi
    if len(valid_items) == 0:
        raise ValueError(f"Không có item nào chứa metric hợp lệ: {score_key}")

    # sắp xếp tăng dần theo score_key
    # điểm càng thấp thì càng đứng trước
    sorted_items = sorted(valid_items, key=lambda x: x[score_key])

    # lấy top_k ảnh có điểm thấp nhất
    return sorted_items[:top_k]

def select_highest_score_items(data: List[Dict[str, Any]], score_key: str = "avg_score",
                               top_k: int = 10) -> List[Dict[str, Any]]:
    '''
    Chọn top_k ảnh có điểm cao nhất theo metric cụ thể
    :param data: danh sách kết quả đọc từ file json 
    :param score_key: tên metric để sắp xếp, mặc định lấy trung bình (avg_score)
    :param top_k: số lượng ảnh cần lấy ra, mặc định là 10
    :return: danh sách top_k items có score cao nhất
    '''

    # lọc ra những item có tồn tại score_key và giá trị không phải None
    valid_items = [
        item for item in data
        if score_key in item and item[score_key] is not None
    ]

    # nếu không có item hợp lệ thì báo lỗi
    if len(valid_items) == 0:
        raise ValueError(f"Không có item nào chứa metric hợp lệ: {score_key}")

    # sắp xếp giảm dần theo score_key
    # điểm càng cao thì càng đứng trước
    sorted_items = sorted(valid_items, key=lambda x: x[score_key], reverse=True)

    # lấy top_k ảnh có điểm cao nhất
    return sorted_items[:top_k]

def plot_caption_from_log(log_item: Dict[str, Any], image_root: Optional[str] = None,
                          save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6), 
                          wrap_width: int = 100) -> None:
    '''
    Vẽ một ảnh cùng với prediction caption, references captions và các metric đánh giá.

    :param log_item: Dictionary chứa thông tin của một ảnh, gồm image_path, file_name, prediction, references và metric
    :param image_root: Thư mục gốc chứa ảnh. Nếu khác None, hàm sẽ dùng image_root + file_name thay vì image_path trong JSON
    :param save_path: Đường dẫn lưu ảnh sau khi vẽ, nếu None thì không lưu
    :param figsize: Kích thước figure khi hiển thị ảnh
    :param wrap_width: Số ký tự tối đa trên mỗi dòng trong phần title
    :return: None
    '''

    # Nếu truyền image_root, ưu tiên tạo đường dẫn ảnh từ image_root và file_name
    if image_root is not None:
        file_name = log_item.get("file_name", None)

        if file_name is None:
            print("Item không có trường file_name.")
            return

        image_path = os.path.join(image_root, file_name)

    # Nếu không truyền image_root, dùng image_path có sẵn trong JSON
    else:
        image_path = log_item.get("image_path", None)

    # Kiểm tra image_path có tồn tại không
    if image_path is None:
        print("Item không có trường image_path.")
        return

    # Kiểm tra file ảnh có tồn tại trong hệ thống không
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return

    # Đọc ảnh và chuyển về RGB
    image = Image.open(image_path).convert("RGB")

    # Lấy prediction caption
    prediction = log_item.get("prediction", "")

    # Lấy danh sách reference captions
    references = log_item.get("references", [])

    # Lấy các metric nếu có
    bleu1 = log_item.get("BLEU-1", None)
    bleu2 = log_item.get("BLEU-2", None)
    bleu3 = log_item.get("BLEU-3", None)
    bleu4 = log_item.get("BLEU-4", None)
    meteor = log_item.get("METEOR", None)
    rouge_l = log_item.get("ROUGE_L", None)
    avg_score = log_item.get("avg_score", None)

    # Khởi tạo title
    title = ""

    # Thêm index và image_id
    title += f"Index: {log_item.get('index', 'N/A')} | "
    title += f"Image ID: {log_item.get('image_id', 'N/A')}\n"

    # Thêm các metric
    if avg_score is not None:
        title += f"Avg Score: {avg_score:.5f} | "

    if bleu1 is not None:
        title += f"BLEU-1: {bleu1:.5f} | "

    if bleu2 is not None:
        title += f"BLEU-2: {bleu2:.5f} | "

    if bleu3 is not None:
        title += f"BLEU-3: {bleu3:.5f} | "

    if bleu4 is not None:
        title += f"BLEU-4: {bleu4:.5f} | "

    if meteor is not None:
        title += f"METEOR: {meteor:.5f} | "

    if rouge_l is not None:
        title += f"ROUGE-L: {rouge_l:.5f}"

    # Thêm prediction
    title += "\n\n"
    title += f"Prediction: {prediction}\n\n"

    # Thêm references
    title += "\n".join([
        f"Ref {i + 1}: {ref}" for i, ref in enumerate(references)
    ])

    # Wrap title cho dễ đọc
    wrapped_title = wrap_multiline_text(text=title, width=wrap_width)

    # Vẽ ảnh
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.title(wrapped_title, fontsize=9, loc="left", pad=12)
    plt.tight_layout()

    # Lưu ảnh nếu có save_path
    if save_path is not None:
        save_folder = os.path.dirname(save_path)

        if save_folder != "":
            os.makedirs(save_folder, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
    plt.close()

def plot_selected_score_images(json_path: str, top_k: int = 10, score_key: str = "avg_score",
                               mode: str = "lowest", image_root: Optional[str] = None, save_dir: Optional[str] = None,
                               figsize: Tuple[int, int] = (8, 6), wrap_width: int = 100) -> None:
    '''
    Đọc file JSON, chọn top_k ảnh tốt nhất hoặc tệ nhất theo một metric,
    sau đó vẽ ảnh với prediction caption và references captions.

    :param json_path: Đường dẫn tới file JSON chứa kết quả đánh giá
    :param top_k: Số lượng ảnh cần vẽ
    :param score_key: Metric dùng để sắp xếp, ví dụ: avg_score, BLEU-4, METEOR
    :param mode: Chế độ chọn ảnh, gồm "lowest" hoặc "highest"
    :param image_root: Thư mục gốc chứa ảnh COCO, dùng để thay thế image_path trong JSON
    :param save_dir: Thư mục lưu ảnh kết quả, nếu None thì chỉ hiển thị và không lưu
    :param figsize: Kích thước figure cho mỗi ảnh
    :param wrap_width: Số ký tự tối đa trên mỗi dòng trong phần title
    :return: None
    '''

    # Đọc dữ liệu từ file JSON
    data = load_json(json_path)

    # Kiểm tra mode
    if mode not in ["lowest", "highest"]:
        raise ValueError("mode phải là 'lowest' hoặc 'highest'.")

    # Chọn ảnh theo mode
    if mode == "lowest":
        selected_items = select_lowest_score_items(data=data, score_key=score_key, top_k=top_k)
        prefix = "lowest"
        print(f"Đã chọn {len(selected_items)} ảnh có {score_key} thấp nhất.")

    else:
        selected_items = select_highest_score_items(data=data, score_key=score_key, top_k=top_k)
        prefix = "highest"
        print(f"Đã chọn {len(selected_items)} ảnh có {score_key} cao nhất.")

    # Tạo thư mục lưu nếu cần
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Duyệt qua từng ảnh được chọn
    for rank, item in enumerate(selected_items, start=1):
        print(
            f"[{rank}/{len(selected_items)}] "
            f"index={item.get('index')} | "
            f"{score_key}={item.get(score_key)} | "
            f"file={item.get('file_name')}"
        )

        # Tạo đường dẫn lưu ảnh nếu có save_dir
        save_path = None

        if save_dir is not None:
            image_id = item.get("image_id", rank)

            save_path = os.path.join(
                save_dir,
                f"{prefix}_{rank:02d}_image_{image_id}.png"
            )

        # Vẽ ảnh
        plot_caption_from_log(log_item=item, image_root=image_root, save_path=save_path, figsize=figsize, wrap_width=wrap_width)

    def show_caption(args, config, vocab, device):
        '''
        Sinh caption cho những ảnh mới 
        '''
        print(f"Đang sinh câu (Architecture: {args.model_type})...")
        vocab_size = len(vocab)
        encoder, decoder = build_model(args, config, vocab_size, args.device)
        # Sinh câu bằng 2 phương pháp
        greedy_cap, img = generate_caption(args.image_path, encoder, decoder, vocab, device, 
                                           model_type=args.model_type, mode='greedy')
        beam_cap, _ = generate_caption(args.image_path, encoder, decoder, vocab, device,
                                        model_type=args.model_type, mode='beam_search', beam_width=5)
        
        # Hiển thị kết quả
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Format tiêu đề cho dễ nhìn
        title_text = f"[{args.model_type.upper()}]\nGreedy: {greedy_cap}\nBeam Search: {beam_cap}"
        plt.title(title_text, fontsize=12, loc='left', pad=10)
        plt.tight_layout()
        plt.show()
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Generating captions for new pics:')
        parser.add_argument("--config",required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--model_type", required=True, choices=['lstm', 'm2_transformer', 'vit_transformer'])
        parser.add_argument("--images_path", required=True)
        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_path = 'vocab.pkl'
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"-> Đã load Vocab với {len(vocab)} từ.")
        config = load_config(args.config)
        show_caption(args, config=config, vocab=vocab, device=device)