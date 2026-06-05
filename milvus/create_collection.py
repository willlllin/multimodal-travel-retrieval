from pymilvus import MilvusClient

client = MilvusClient(
    "D:/travel-agent/milvus/travel.db"
)

client.create_collection(
    collection_name="travel_images",
    dimension=1024
)

print(client.list_collections())