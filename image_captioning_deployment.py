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


def wrap_text(text, width=90):
    '''
    Chỉnh text của plt
    '''
    return "\n".join(textwrap.wrap(text, width=width))


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
            title = (
                f"File: {file_name}\n"
                f"Greedy: {greedy_cap}\n"
                f"Beam Search: {beam_cap}"
            )

            plt.figure(figsize=(7, 7))
            plt.imshow(image)
            plt.axis("off")
            plt.title(wrap_text(title, width=90), fontsize=10, loc="left", pad=10)
            plt.tight_layout()
            plt.show()

            save_dir = "experiments/deployment_outputs"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{os.path.splitext(file_name)[0]}_caption.png"
            )

            plt.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close()

            print(f"Đã lưu ảnh kết quả tại: {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating captions for new images from folder")

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=["lstm", "m2_transformer", "vit_transformer"])
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--vocab_path", required=True, default="vocab.pkl")
    parser.add_argument("--beam_size", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    transform = get_transforms()

    results = caption_images_from_folder(args=args, config=config, vocab=vocab,
                                         transform=transform, device=device, show_image=True)