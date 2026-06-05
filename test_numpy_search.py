import numpy as np

DIM = 768

# 模拟图片库向量
image_vectors = np.random.rand(10, DIM).astype("float32")

# 归一化
image_vectors = image_vectors / np.linalg.norm(image_vectors, axis=1, keepdims=True)

# 模拟 query 向量
query_vector = np.random.rand(1, DIM).astype("float32")
query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

# cosine similarity
scores = query_vector @ image_vectors.T

top_k = 5
top_indices = np.argsort(scores[0])[::-1][:top_k]

for rank, idx in enumerate(top_indices, 1):
    print(f"Top {rank}: image_{idx}, score={scores[0][idx]:.4f}")