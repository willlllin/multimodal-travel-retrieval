import sys
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

PROJECT_ROOT = Path("D:/travel-agent")
MAGICLENS_ROOT = PROJECT_ROOT / "magiclens" / "magiclens"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_SIZE = "large"
MODEL_PATH = str(MAGICLENS_ROOT / "models" / "magic_lens_clip_large.pkl")

QUERY_IMAGE = PROJECT_ROOT / "test.jpg"
INSTRUCTION = "find similar places with fewer tourists"
TOP_K = 5

sys.path.append(str(MAGICLENS_ROOT))

from inference import load_model
from data_utils import process_img
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


def encode_query(model, model_params, tokenizer, image_path, instruction):
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

    emb = np.array(outputs["multimodal_embed_norm"][0]).astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def main():
    vectors = np.load(OUTPUT_DIR / "image_vectors.npy")

    with open(OUTPUT_DIR / "image_paths.json", "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    print("加载图片库向量:", vectors.shape)
    print("加载 MagicLens 模型...")

    tokenizer = clip_tokenizer.build_tokenizer()
    model, model_params = load_model(
        model_size=MODEL_SIZE,
        model_path=MODEL_PATH,
    )

    query_vector = encode_query(
        model=model,
        model_params=model_params,
        tokenizer=tokenizer,
        image_path=QUERY_IMAGE,
        instruction=INSTRUCTION,
    )

    scores = query_vector @ vectors.T
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print("\n检索结果：")
    for rank, idx in enumerate(top_indices, 1):
        print(f"Top {rank}: score={scores[idx]:.4f}")
        print(image_paths[idx])


if __name__ == "__main__":
    main()