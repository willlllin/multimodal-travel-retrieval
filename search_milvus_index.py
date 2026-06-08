import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from pymilvus import connections, Collection

# =========================
# 1. 路径与基础配置
# =========================

PROJECT_ROOT = Path("D:/travel-agent")
MAGICLENS_ROOT = PROJECT_ROOT / "magiclens" / "magiclens"

MODEL_SIZE = "large"
MODEL_PATH = str(MAGICLENS_ROOT / "models" / "magic_lens_clip_large.pkl")

COLLECTION_NAME = "travel_images"

QUERY_IMAGE = PROJECT_ROOT / "test.jpg"
INSTRUCTION = "find similar places with fewer tourists"
TOP_K = 5

# 作用：
# 让 Python 可以 import MagicLens 源码中的 inference.py / data_utils.py
sys.path.append(str(MAGICLENS_ROOT))

from inference import load_model
from data_utils import process_img
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


# =========================
# 2. 编码 Query 图片 + 指令
# =========================

def encode_query(model, model_params, tokenizer, image_path: Path, instruction: str):
    """
    输入：
        query 图片路径
        文本指令

    输出：
        768维 query embedding
    """

    # 作用：
    # 使用 MagicLens 官方 process_img 处理图片
    # 输出 shape: (1, 224, 224, 3)
    image = process_img(str(image_path), 224)

    # 作用：
    # 使用 CLIP tokenizer 把文本指令变成 token ids
    tokens = np.array(tokenizer(instruction))

    # 作用：
    # 确保 tokens 是 batch 形式
    # 从 (77,) 变成 (1, 77)
    if tokens.ndim == 1:
        tokens = tokens[None, :]

    # 作用：
    # 调用 MagicLens 生成多模态 embedding
    outputs = model.apply(
        model_params,
        {
            "ids": jnp.array(tokens),
            "image": jnp.array(image),
        },
    )

    # 作用：
    # 取归一化后的多模态向量
    # shape: (768,)
    embedding = np.array(outputs["multimodal_embed_norm"][0]).astype("float32")

    # 作用：
    # 再保险做一次 L2 normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

    return embedding


# =========================
# 3. Milvus 检索
# =========================

def search_milvus(query_embedding: np.ndarray, top_k: int = 5):
    """
    输入：
        query embedding

    输出：
        Milvus TopK 检索结果
    """

    # 作用：
    # 连接本地 Milvus 服务
    # 前提：docker compose up -d 后 milvus-standalone 正在运行
    connections.connect(
        alias="default",
        host="127.0.0.1",
        port="19530",
    )

    # 作用：
    # 获取已经创建好的 travel_images collection
    collection = Collection(COLLECTION_NAME)

    # 作用：
    # 确保 collection 已经加载到内存
    collection.load()

    # 作用：
    # 调用 Milvus 做向量检索
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="vector",
        param={
            "metric_type": "COSINE",
            "params": {},
        },
        limit=top_k,
        output_fields=["image_path"],
    )

    return results


# =========================
# 4. 主流程
# =========================

def main():
    print("加载 MagicLens tokenizer...")
    tokenizer = clip_tokenizer.build_tokenizer(
    bpe_path=r"D:\travel-agent\CLIP\clip\bpe_simple_vocab_16e6.txt.gz"
)

    print("加载 MagicLens 模型...")
    model, model_params = load_model(
        model_size=MODEL_SIZE,
        model_path=MODEL_PATH,
    )

    print("生成 Query Embedding...")
    query_embedding = encode_query(
        model=model,
        model_params=model_params,
        tokenizer=tokenizer,
        image_path=QUERY_IMAGE,
        instruction=INSTRUCTION,
    )

    print("Query Embedding shape:", query_embedding.shape)

    print("连接 Milvus 并执行搜索...")
    results = search_milvus(
        query_embedding=query_embedding,
        top_k=TOP_K,
    )

    print("\n检索结果：")

    for rank, hit in enumerate(results[0], start=1):
        image_path = hit.entity.get("image_path")
        score = hit.score

        print(f"Top {rank}")
        print(f"Score: {score:.4f}")
        print(f"Image: {image_path}")
        print("-" * 50)


if __name__ == "__main__":
    main()