import os
import sys
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

# ===== 路径配置 =====
PROJECT_ROOT = Path("D:/travel-agent")
MAGICLENS_ROOT = PROJECT_ROOT / "magiclens" / "magiclens"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_SIZE = "large"
MODEL_PATH = str(MAGICLENS_ROOT / "models" / "magic_lens_clip_large.pkl")
INSTRUCTION_FOR_INDEX = ""  # 入库图片使用空指令
DIM = 768

# 让 Python 找到 MagicLens 源码
sys.path.append(str(MAGICLENS_ROOT))

from inference import load_model
from data_utils import process_img
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


def find_images(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in exts:
                image_paths.append(Path(root) / file)
    return image_paths


def encode_one_image(model, model_params, tokenizer, image_path: Path, instruction: str):
    image = process_img(str(image_path), 224)

    tokens = np.array(tokenizer(instruction))
    if tokens.ndim == 1:
        tokens = tokens[None, :]

    outputs = model.apply(
        model_params,
        {
            "ids": jnp.array(tokens),
            "image": jnp.array(image),
        },
    )

    embedding = np.array(outputs["multimodal_embed_norm"][0]).astype("float32")
    return embedding


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(IMAGE_DIR)

    if not image_paths:
        raise ValueError(f"没有在 {IMAGE_DIR} 找到图片，请先放入 jpg/png/webp 图片。")

    print(f"找到图片数量: {len(image_paths)}")
    print("加载 MagicLens 模型...")

    tokenizer = clip_tokenizer.build_tokenizer()
    model, model_params = load_model(
        model_size=MODEL_SIZE,
        model_path=MODEL_PATH,
    )

    vectors = []
    valid_paths = []

    for image_path in tqdm(image_paths, desc="Encoding images"):
        try:
            emb = encode_one_image(
                model=model,
                model_params=model_params,
                tokenizer=tokenizer,
                image_path=image_path,
                instruction=INSTRUCTION_FOR_INDEX,
            )

            if emb.shape[0] != DIM:
                raise ValueError(f"向量维度错误: {emb.shape}, expected {DIM}")

            vectors.append(emb)
            valid_paths.append(str(image_path).replace("\\", "/"))

        except Exception as e:
            print(f"[跳过] {image_path}，原因: {e}")

    if not vectors:
        raise RuntimeError("没有成功生成任何图片向量。")

    vectors = np.vstack(vectors).astype("float32")

    # 理论上 MagicLens 已经输出归一化向量，这里再保险归一化一次
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    np.save(OUTPUT_DIR / "image_vectors.npy", vectors)

    with open(OUTPUT_DIR / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(valid_paths, f, ensure_ascii=False, indent=2)

    print("索引构建完成")
    print("向量文件:", OUTPUT_DIR / "image_vectors.npy")
    print("路径文件:", OUTPUT_DIR / "image_paths.json")
    print("向量 shape:", vectors.shape)


if __name__ == "__main__":
    main()