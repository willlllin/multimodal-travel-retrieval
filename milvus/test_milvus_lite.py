from pymilvus import MilvusClient

client = MilvusClient(
    "D:/travel-agent/milvus/travel.db"
)

print("Milvus Lite启动成功")
print(client.list_collections())