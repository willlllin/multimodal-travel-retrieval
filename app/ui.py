import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path("D:/travel-agent")
MAGICLENS_ROOT = PROJECT_ROOT / "magiclens" / "magiclens"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_SIZE = "large"
MODEL_PATH = str(MAGICLENS_ROOT / "models" / "magic_lens_clip_large.pkl")
TOP_K = 5

sys.path.append(str(MAGICLENS_ROOT))

from inference import load_model
from data_utils import process_img
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


@st.cache_resource
def load_magiclens():
    tokenizer = clip_tokenizer.build_tokenizer()
    model, model_params = load_model(
        model_size=MODEL_SIZE,
        model_path=MODEL_PATH,
    )
    return tokenizer, model, model_params


@st.cache_resource
def load_index():
    vectors = np.load(OUTPUT_DIR / "image_vectors.npy")

    with open(OUTPUT_DIR / "image_paths.json", "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    return vectors, image_paths


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


def search(query_vector, vectors, top_k=5):
    scores = query_vector @ vectors.T
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(idx, float(scores[idx])) for idx in top_indices]


def main():
    st.set_page_config(
        page_title="Multi-modal Travel Search Agent V1",
        layout="wide",
    )

    st.title("Multi-modal Travel Search Agent V1")
    st.write("上传一张旅行图片，并输入检索指令，系统会返回图片库中最匹配的 TopK 结果。")

    uploaded_file = st.file_uploader(
        "上传 Query 图片",
        type=["jpg", "jpeg", "png", "webp"],
    )

    instruction = st.text_input(
        "输入检索指令",
        value="find similar places with fewer tourists",
    )

    top_k = st.slider("返回数量 TopK", 1, 10, TOP_K)

    if uploaded_file is not None:
        st.subheader("Query 图片")
        st.image(uploaded_file, width=300)

    if st.button("Search"):
        if uploaded_file is None:
            st.warning("请先上传一张图片。")
            return

        with st.spinner("加载模型和图片索引..."):
            tokenizer, model, model_params = load_magiclens()
            vectors, image_paths = load_index()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            query_image_path = tmp.name

        with st.spinner("正在生成 Query Embedding 并检索..."):
            query_vector = encode_query(
                model=model,
                model_params=model_params,
                tokenizer=tokenizer,
                image_path=query_image_path,
                instruction=instruction,
            )

            results = search(
                query_vector=query_vector,
                vectors=vectors,
                top_k=top_k,
            )

        st.subheader("检索结果")

        cols = st.columns(min(top_k, 5))

        for rank, (idx, score) in enumerate(results, 1):
            image_path = image_paths[idx]
            col = cols[(rank - 1) % len(cols)]

            with col:
                st.image(image_path, use_container_width=True)
                st.write(f"Top {rank}")
                st.write(f"Score: {score:.4f}")
                st.caption(image_path)


if __name__ == "__main__":
    main()