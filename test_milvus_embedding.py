from pymilvus import MilvusClient
import numpy as np

DB_PATH = "D:/travel-agent/milvus/travel.db"
COLLECTION_NAME = "travel_images"
DIM = 768

client = MilvusClient(DB_PATH)

if COLLECTION_NAME in client.list_collections():
    client.drop_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=DIM,
)

vector = np.random.rand(DIM).astype("float32").tolist()

client.insert(
    collection_name=COLLECTION_NAME,
    data=[
        {
            "id": 1,
            "vector": vector,
            "image_path": "D:/travel-agent/data/00bbb1595034fc58685b8aebaf7b16bc.jpeg",
            "instruction": "find similar places"
        }
    ]
)

results = client.search(
    collection_name=COLLECTION_NAME,
    data=[vector],
    limit=1,
    output_fields=["image_path", "instruction"]
)

print(results)