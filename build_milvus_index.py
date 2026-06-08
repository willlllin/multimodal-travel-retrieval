from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections
import numpy as np
from pathlib import Path
import json

# 连接 Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Collection 配置
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500)
]
schema = CollectionSchema(fields, description="Travel image embeddings")
collection_name = "travel_images"

from pymilvus import utility

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# 创建 Collection
collection = Collection(name=collection_name, schema=schema, using="default", shards_num=1)
print(f"Collection '{collection_name}' 创建成功")

# 加载之前生成的 embedding
output_dir = Path("D:/travel-agent/outputs")
vectors = np.load(output_dir / "image_vectors.npy")
with open(output_dir / "image_paths.json", "r", encoding="utf-8") as f:
    image_paths = json.load(f)

# 插入 Milvus
collection.insert([vectors.tolist(), image_paths])
collection.flush()

index_params = {
    "metric_type": "COSINE",
    "index_type": "FLAT",
    "params": {}
}

collection.create_index(
    field_name="vector",
    index_params=index_params
)

collection.load()
print("向量数据插入完成并加载到 Milvus")