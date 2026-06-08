# Multi-modal Travel Retrieval System

基于 MagicLens 与 Milvus 的多模态旅行图片检索系统。

## 项目简介

Multi-modal Travel Retrieval System 是一个面向旅游场景的多模态检索系统。

用户可以上传一张旅行图片，并结合自然语言指令进行检索，例如：

```text
上传：富士山图片

指令：
find similar places with fewer tourists
```

系统将利用 MagicLens 生成图像与文本联合语义向量，在旅行图片库中完成 Top-K 相似检索，并返回最匹配的景点图片。

当前版本实现了从图片上传、Embedding 生成、向量存储、向量检索到结果展示的完整闭环。

---

## 系统架构

```text
Query Image
      +
 Natural Language Instruction
      |
      v
  MagicLens
(Multimodal Encoder)
      |
      v
768-d Embedding
      |
      v
Milvus Vector Database
      |
      v
Top-K Retrieval
      |
      v
Streamlit UI
```

---

## 技术栈

### Retrieval Model

* MagicLens (Google DeepMind)
* CLIP ViT-L/14 Backbone
* JAX
* Flax

### Vector Database

* Milvus 2.4.4
* Etcd
* MinIO

### Frontend

* Streamlit

### Data Processing

* Pillow
* NumPy

### Deployment

* Docker Desktop
* Docker Compose

---

## 核心功能

### 多模态检索

支持：

```text
Image + Instruction
```

联合检索。

示例：

```text
find similar places

find similar beaches

find similar places with fewer tourists

find similar places in winter
```

---

### 图片向量化

使用 MagicLens 将图片与文本映射至统一语义空间。

输出：

```text
768-dimensional embedding
```

用于后续向量检索。

---

### Milvus 向量检索

使用 Milvus 存储和管理图片向量。

当前索引：

```text
FLAT
COSINE Similarity
```

支持：

```text
Top-K Similarity Search
```

后续可扩展：

```text
HNSW
IVF_FLAT
IVF_SQ8
```

以支持十万级以上图片库。

---

### Web 可视化界面

基于 Streamlit 提供交互界面：

* 图片上传
* 检索指令输入
* Top-K 结果展示
* 相似度分数展示

---

## 项目结构

```text
travel-agent/

├── app/
│   └── ui.py

├── data/
│   └── images/

├── outputs/

├── milvus/
│   ├── docker-compose.yml
│   └── volumes/

├── magiclens/

├── build_milvus_index.py

├── search_milvus_index.py

└── README.md
```

---

## 环境要求

### Python

```text
Python 3.9+
```

### Docker

```text
Docker Desktop
```

### Milvus

```text
Milvus 2.4.4
Etcd
MinIO
```

---

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/willlllin/multimodal-travel-retrieval.git

cd multimodal-travel-retrieval
```

---

### 2. 创建虚拟环境

```bash
python -m venv travel

travel\Scripts\activate
```

---

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

### 4. 启动 Milvus

进入：

```bash
cd milvus
```

启动：

```bash
docker compose up -d
```

查看：

```bash
docker ps
```

应看到：

```text
milvus-standalone
milvus-etcd
milvus-minio
```

---

### 5. 构建向量索引

```bash
python build_milvus_index.py
```

该步骤将：

1. 加载 MagicLens
2. 生成图片 Embedding
3. 写入 Milvus
4. 创建向量索引

---

### 6. 启动系统

```bash
python -m streamlit run app/ui.py
```

浏览器访问：

```text
http://localhost:8501
```

---

## 当前版本

### V1

```text
MagicLens
+
NumPy Search
+
Streamlit
```

完成：

* 图片向量生成
* NumPy Top-K 检索

---

### V2

```text
MagicLens
+
Milvus
+
Streamlit
```

完成：

* Milvus 部署
* Collection 管理
* 向量存储
* 向量索引构建
* Top-K 检索
* Web UI 集成

---

## 后续规划

### V3：Metadata Retrieval

引入：

```text
Country
City
Season
Category
```

实现：

```text
Vector Search
+
Metadata Filtering
```

---

### V4：Travel Recommendation Agent

结合：

```text
Image Search
+
User Preference
+
Budget Constraint
+
Trip Planning
```

实现旅行推荐与行程规划能力。

---

## 致谢

* Google DeepMind MagicLens
* Scenic
* CLIP
* Milvus Community
* Streamlit
