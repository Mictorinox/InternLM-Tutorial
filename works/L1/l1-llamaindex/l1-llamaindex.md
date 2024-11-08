本文为笔者参与书生大模型实战营第四期的关卡任务实现记录。在此仅作流程演示和简要说明。如果想了解更多关于原理和不同实现方式的细节，强烈推荐参考主办方提供的[说明文档](https://github.com/InternLM/Tutorial/blob/camp4/docs/L1/LlamaIndex/readme.md)。

# 课程任务

l1-Llamaindex 课程任务如下：
基于 LlamaIndex 构建自己的 RAG 知识库，寻找一个问题 A 在使用 LlamaIndex 之前 InternLM2-Chat-1.8B 模型不会回答，借助 LlamaIndex 后 InternLM2-Chat-1.8B 模型具备回答 A 的能力.

本文小标题如下

1. 环境配置
2. baseline调试
3. RAG调试

# 1. 环境配置

## 1.1 baseline

pytorch：pytorch, torchvision, torchaudio, torch-cuda

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

einop 库用于张量操作，protobuf 库提供轻量化的二进制数据存储格式

```
pip install einops==0.7.0 protobuf==5.26.1
```

llamaindex 库是一套构建上下文增强 LLM 的框架，是本次 RAG 工作的核心
[llamaindex 文档](https://llama-index.readthedocs.io/zh/latest/index.html)

```
pip install llama-index==0.10.38 llama-index-llms-huggingface==0.2.0 "transformers[torch]==4.41.1" "huggingface_hub[inference]==0.23.1" huggingface_hub==0.23.1
```

## 1.2 RAG Specific

nltk 库提供了包括分词、词汇规范化等多项操作，并提供了多个语料库（古腾堡、布朗、路透社、就职演说等）
国内安装需要先从镜像下载 nltk_data 库[github 链接](https://github.com/nltk/nltk_data)，将其中的 packages 文件夹重命名为 nltk_data,然后移动到环境变量 path 路径下。然后解压其中的 tokenizers/punkt.zip 和 taggers/perceptron_tagger.zip

```
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
```

安装 llama-index-embeddings，这是 llamaindex 的嵌入算法包

```
pip install llama-index-embeddings-huggingface==0.2.0 llama-index-embeddings-instructor==0.1.3
```
这步会卸载 torch=2.0.1 并重新安装 torch=2.5.1，导致 bug
解决方法：重新安装 torch
```
conda install --force-reinstall pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

安装 sentence-transformers，这是本次用于检索和嵌入的词向量模型
```
sentence-transformers==2.7.0 sentencepiece==0.2.0
```

# 2. baseline调试

llamaindex 提供了相对简便的框架，使用 HuggingFaceLLM 类实现的代码如下：

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage

llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)

rsp = llm.chat(messages=[ChatMessage(content="谁在2024年美国总统大选中获胜了？")])
print(rsp)
```

此时尝试问 baseline 模型谁在 2024 年美国大选中获胜，模型是无法回答的：
![alt text](images/image.png)

# 3. RAG调试

## 3.1 指定词向量模型

词向量模型在 rag 环节中用于检索和嵌入，demo 使用的是 sentence-transformers。

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings

embed_model = HuggingFaceEmbedding(
    model_name="/root/model/sentence-transformer"
)
Settings.embed_model = embed_model
```

## 3.2 设置全局的 llm 属性

```python
from llama_index.llms.huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
Settings.llm = llm
```

## 3.3 构建知识库

我们引入相关新闻报道作为知识库：
![alt text](images/image-2.png)
从指定目录读取文档：

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
```

建立文档索引：

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
```
使用文档索引建立查询引擎，并进行查询：

```python
query_engine = index.as_query_engine()
response = query_engine.query("谁在2024年美国总统大选中获胜了？")
```

模型可以正确回答了:

![alt text](images/image-1.png)
