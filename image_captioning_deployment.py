import argparse
import os
import pickle
import textwrap
import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.evaluation.eval import build_model
from inference import generate_caption
from src.data.dataset import get_transforms
from src.utils import load_config


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

def caption_images_from_folder(args, config, vocab, transform, device, show_image=True):
    """
    Sinh caption cho các ảnh mới trong một folder, do người dùng tự nhập
    """
    vocab_size = len(vocab)

    encoder, decoder = build_model(args=args, config=config, vocab_size=vocab_size,device=device)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    image_files = [
        f for f in os.listdir(args.folder_path)
        if f.lower().endswith(image_extensions)
    ]

    if len(image_files) == 0:
        print("Không tìm thấy ảnh trong folder.")
        return []

    results = []

    for file_name in image_files:
        image_path = os.path.join(args.folder_path, file_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Lỗi đọc ảnh {file_name}: {e}")
            continue

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            greedy_cap = generate_caption(image_tensor, encoder, decoder, vocab, 
                                          device, model_type=args.model_type, mode="greedy")

            beam_cap = generate_caption(image_tensor, encoder, decoder, vocab, device, 
                                        model_type=args.model_type, mode="beam_search", beam_width=args.beam_size)

        captions = {
            "greedy": greedy_cap,
            "beam_search": beam_cap
        }

        results.append({
            "file_name": file_name,
            "image_path": image_path,
            "captions": captions
        })

        print(f"Ảnh: {file_name}")
        print(f"Greedy: {greedy_cap}")
        print(f"Beam Search: {beam_cap}")
        print("-" * 80)

        if show_image:
            save_dir = "experiments/deployment_outputs"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{os.path.splitext(file_name)[0]}_caption.png"
            )

            title = ''
            title += f"File name: {file_name}\n"
            title += f"Greedy Search: {greedy_cap}\n"
            title += f"Beam Search: {beam_cap}"
            wrapped_title = wrap_multiline_text(text=title, width=90)

            plt.figure(figsize=(8,6))
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
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating captions for new images from folder")

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=["lstm", "m2_transformer", "vit_transformer"])
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--vocab_path", default="vocab.pkl")
    parser.add_argument("--beam_size", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    transform = get_transforms()

    results = caption_images_from_folder(args=args, config=config, vocab=vocab,
                                         transform=transform, device=device, show_image=True)