import sys
import tempfile
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import streamlit as st
from pymilvus import connections, Collection

PROJECT_ROOT = Path("D:/travel-agent")
MAGICLENS_ROOT = PROJECT_ROOT / "magiclens" / "magiclens"

MODEL_SIZE = "large"
MODEL_PATH = r"D:\travel-agent\magiclens\magiclens\models\magic_lens_clip_large.pkl"
BPE_PATH = r"D:\travel-agent\CLIP\clip\bpe_simple_vocab_16e6.txt.gz"

COLLECTION_NAME = "travel_images"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

DEFAULT_TOP_K = 5

sys.path.append(str(MAGICLENS_ROOT))

from inference import load_model
from data_utils import process_img
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


@st.cache_resource
def load_magiclens():
    """
    作用：
    加载 MagicLens tokenizer、模型结构和模型权重。
    Streamlit 会缓存这个结果，避免每次点击 Search 都重新加载模型。
    """
    tokenizer = clip_tokenizer.build_tokenizer(
        bpe_path=BPE_PATH
    )

    model, model_params = load_model(
        model_size=MODEL_SIZE,
        model_path=MODEL_PATH,
    )

    return tokenizer, model, model_params


@st.cache_resource
def connect_milvus():
    """
    作用：
    连接本地 Milvus 服务，并加载 travel_images collection。
    前提：
    Docker 中 milvus-standalone 正在运行。
    """
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )

    collection = Collection(COLLECTION_NAME)
    collection.load()

    return collection


def encode_query(model, model_params, tokenizer, image_path: str, instruction: str):
    """
    输入：
    - 用户上传的 query 图片
    - 用户输入的自然语言指令

    输出：
    - 768维 MagicLens query embedding
    """
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

    embedding = np.array(
        outputs["multimodal_embed_norm"][0]
    ).astype("float32")

    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

    return embedding


def search_milvus(collection, query_embedding, top_k):
    """
    作用：
    使用 Milvus 对 query embedding 进行 TopK 向量检索。
    """
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


def main():
    st.set_page_config(
        page_title="Multi-modal Travel Retrieval System",
        layout="wide",
    )

    st.title("Multi-modal Travel Retrieval System")
    st.caption("MagicLens + Milvus + Streamlit")

    st.write(
        "上传一张旅行图片，并输入检索指令，系统会基于 MagicLens 生成多模态向量，"
        "再通过 Milvus 返回最相似的 Top-K 图片。"
    )

    uploaded_file = st.file_uploader(
        "上传 Query 图片",
        type=["jpg", "jpeg", "png", "webp"],
    )

    instruction = st.text_input(
        "输入检索指令",
        value="find similar places with fewer tourists",
    )

    top_k = st.slider(
        "返回数量 TopK",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    if uploaded_file is not None:
        st.subheader("Query 图片")
        st.image(uploaded_file, width=320)

    if st.button("Search"):
        if uploaded_file is None:
            st.warning("请先上传一张图片。")
            return

        with st.spinner("加载 MagicLens 模型和 Milvus Collection..."):
            tokenizer, model, model_params = load_magiclens()
            collection = connect_milvus()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            query_image_path = tmp.name

        with st.spinner("正在生成 Query Embedding 并检索..."):
            query_embedding = encode_query(
                model=model,
                model_params=model_params,
                tokenizer=tokenizer,
                image_path=query_image_path,
                instruction=instruction,
            )

            results = search_milvus(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
            )

        st.subheader("检索结果")

        hits = results[0]

        if len(hits) == 0:
            st.warning("没有检索到结果。")
            return

        cols = st.columns(min(top_k, 5))

        for rank, hit in enumerate(hits, start=1):
            image_path = hit.entity.get("image_path")
            score = hit.score

            col = cols[(rank - 1) % len(cols)]

            with col:
                st.image(image_path, use_container_width=True)
                st.markdown(f"**Top {rank}**")
                st.write(f"Score: {score:.4f}")
                st.caption(image_path)


if __name__ == "__main__":
    main()