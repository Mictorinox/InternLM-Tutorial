本文为笔者参与书生大模型实战营第四期的关卡任务实现记录。在此仅作流程演示和简要说明。如果想了解更多关于原理和不同实现方式的细节，强烈推荐参考主办方提供的[说明文档](https://github.com/InternLM/Tutorial/blob/camp4/docs/L2/LMDeploy/readme.md)。

# 前言

本地部署大模型是开发大模型应用的必备前置工作。为提高大模型在本地的训练和推理性能，可以通过模型量化缩短参数精度。本文实现了使用上海人工智能实验室开发的 lmdeploy 框架在本地量化和部署 internlm 2.5 1.8B 模型，并通过 OpenAI库的函数调用接口实现了让大模型调用函数。

使用结合 W4A16 量化与 kv cache 量化的`internlm2_5-1_8b-chat`模型封装本地 API 并与大模型进行一次对话，作业截图需包括显存占用情况与大模型回复，参考 4.1 API 开发，**请注意 2.2.3 节与 4.1 节应使用作业版本命令。**
使用 Function call 功能让大模型完成一次简单的"加"与"乘"函数调用，作业截图需包括大模型回复的工具调用情况，参考 4.2 Function call(选做)

# 简单部署流程

## 配置环境

```sh
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install timm==1.0.8 openai==1.40.3 lmdeploy[all]==0.5.3

pip install datasets==2.19.2
```

## 获取模型

lmstudio 可以运行多种格式的模型
官方文档提供了[支持的模型列表](https://lmdeploy.readthedocs.io/zh-cn/latest/supported_models/supported_models.html)

实战营的开发环境提供了下载好的模型：

```sh
mkdir /root/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat /root/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat /root/models
ln -s /root/share/new_models/OpenGVLab/InternVL2-26B /root/models
```

## 验证模型文件

我们可以通过尝试启动对话服务，以此验证模型文件是否可以正常工作。

```sh
lmdeploy chat /root/models/internlm2_5-1_8b-chat
```

![alt text](images/image.png)

# 通过 api 形式部署模型

```sh
lmdeploy serve api_server \
    /root/models/internlm2_5-1_8b-chat \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 45678 \
    --tp 1
```

其中`--tp 1`参数表示并行 GPU 数量

![alt text](images/image-1.png)

进行端口转发后可以在本地浏览器打开，说明部署成功。

![alt text](images/image-2.png)

此时可以新建一个进程，使用`nvidia-smi`指令查看显存占用情况，本文是在书生实战营提供的环境，因此使用`studio-smi`。

![alt text](images/image-5.png)

在此可以看到，当前显存占用 20808MB

## 用命令行连接到 API 服务器

启动 api 服务器后，可以通过`lmdeploy serve api_client`命令连接到服务器，开启对话

```sh
lmdeploy serve api_client http://localhost:45678
```

![alt text](images/image-3.png)

输出结果和直接在命令行部署是一样的。

## 用 Gradio 网页连接到 API 服务器

lmdeploy 提供了一个简单的 gradio 界面，用于展示模型。同时，官网文档还提供了在 huggingface 上创建模型的在线 demo 的教程。[部署 gradio 服务 — lmdeploy 官方文档](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/gradio.html)

我们通过命令`lmdeploy serve gradio`启动 gradio 服务：

```sh
lmdeploy serve gradio http://localhost:45678 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

转发 gradio 服务器的端口 6006 并在本地打开：

![alt text](images/image-4.png)

gradio 界面下方的拖动条可以快速调整参数。关于大模型的 top_p 和 temperature 参数，简单地说，top_p 取值在 0-1 之间，越接近 1 模型的备选项会越多；temperature 理论取值范围是 0-正无穷，在此最大可以取 1.5，取值越大模型越倾向于随机选择备选项（而不是高分备选项）。感兴趣的同学可以参考这篇文章：[大模型文本生成——解码策略（Top-k & Top-p & Temperature）](https://www.zhihu.com/tardis/zm/art/647813179)

# 提升部署性能

## 设置 kv cache

kv cache 是什么？

```sh
lmdeploy serve api_server \
    /root/models/internlm2_5-1_8b-chat \
    --model-format hf \
    --quant-policy 4 \
    --cache-max-entry-count 0.4\
    --server-name 0.0.0.0 \
    --server-port 45678 \
    --tp 1
```

`quant_policy` 参数设置为 4 表示 kv int4 量化，如果设置为 8 表示 kv int8 量化
`cachemax-entry-count`参数表示 kv cache 大小占剩余显存的比例，默认为 0.8

我们同样监测显存使用情况
![alt text](images/image-6.png)

可以看到，在设置 kv cache 占比后显存占用为 12616MB。相比未设置时占用 20808MB 有了较大优化。

优化的大小可以通过简单的估算验证：

1、在 BF16 精度下，1.8B 模型权重占用**3.6GB**：18×10^9 parameters×2 Bytes/parameter=**3.6GB**

2、kv cache 占用**16.32GB**：剩余显存**24-3.6=20.4GB**，kv cache 默认占用 80%，即**20.4\*0.8=16.32GB**

3、由此反推计算其他项大约**0.88GB**：实际占用**20.8GB**-权重占用**3.6GB**-kv cache 占用**16.32GB**=0.88GB。

对于修改 kv cache 占用之后的显存占用情况(**12.6GB**)：

1、与上述声明一致，在 BF16 精度下，1.8B 模型权重占用**3.6GB**

2、kv cache 占用**4GB**：剩余显存**24-3.6=20.4GB**，kv cache 修改为占用 40%，即**20.4\*0.4=8.16GB**

3、其他项**0.88GB**

由此估算 总占用显存=权重占用**14GB**+kv cache 占用**4GB**+其它项**0.88GB** = 12.64GB，与实际占用基本一致。

## 模型量化

模型量化是一种优化技术，可以减少模型大小并提高推理速度。主要通过调整模型的权重变量和激活变量的长度实现，例如 W4A16 量化表示权重变量用 4 位整数表示，激活变量用 16 位浮点数表示。

```sh
lmdeploy lite auto_awq \
   /root/models/internlm2_5-1_8b-chat \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w-group-size 128 \
  --batch-size 1 \
  --search-scale False \
  --work-dir /root/models/internlm2_5-1_8b-chat-w4a16-4bit
```

`lite`指令由于启动量化任务，参数 auto_awq 表示自动权重量化
`--calib-dataset`用于指定校准数据集，`--calib-samples`指定校准样本数量，`--calib-swqlen 2048`用于指定校准过程的序列长度
`--w-bits`指定权重的位数

在此如果报错：`TypeError: 'NoneType' object is not callable`，可能的原因是当前版本的 datasets3.0 无法下载 calibrate 数据集通过`pip install datasets==2.19.2` 可以解决。

# 使用本地部署模型

## API 访问

启动 api 服务器

```sh
lmdeploy serve api_server \
    /root/models/internlm2_5-1_8b-chat-w4a16-4bit \
    --model-format awq \
    --cache-max-entry-count 0.4 \
    --quant-policy 4 \
    --server-name 0.0.0.0 \
    --server-port 45678 \
    --tp 1
```

新建 `internlm2_5.py`，内容如下：

```py
# 导入openai模块中的OpenAI类，这个类用于与OpenAI API进行交互
from openai import OpenAI


# 创建一个OpenAI的客户端实例，需要传入API密钥和API的基础URL
client = OpenAI(
    api_key='YOUR_API_KEY',
    # 替换为你的OpenAI API密钥，由于我们使用的本地API，无需密钥，任意填写即可
    base_url="http://0.0.0.0:45678/v1"
    # 指定API的基础URL，这里使用了本地地址和端口
)

# 调用client.models.list()方法获取所有可用的模型，并选择第一个模型的ID
# models.list()返回一个模型列表，每个模型都有一个id属性
model_name = client.models.list().data[0].id

# 使用client.chat.completions.create()方法创建一个聊天补全请求
# 这个方法需要传入多个参数来指定请求的细节
response = client.chat.completions.create(
  model=model_name,
  # 指定要使用的模型ID
  messages=[
  # 定义消息列表，列表中的每个字典代表一个消息
    {"role": "system", "content": "你是一个友好的小助手，负责解决问题."},
    # 系统消息，定义助手的行为
    {"role": "user", "content": "帮我讲述一个关于狐狸和西瓜的小故事"},
    # 用户消息，询问时间管理的建议
  ],
    temperature=0.8,
    # 控制生成文本的随机性，值越高生成的文本越随机
    top_p=0.8
    # 控制生成文本的多样性，值越高生成的文本越多样
)

# 打印出API的响应结果
print(response.choices[0].message.content)
```

`python internlm2_5.py`

## Function call

利用大模型调用函数，根据用户提问输出参数执行函数，并将函数输出结果作为回答问题的依据

新建`internlm2_5_func.py`，内容如下：

```py
from openai import OpenAI


def add(a: int, b: int):
    return a + b


def mul(a: int, b: int):
    return a * b


tools = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "use this function to calculate the addition of two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "int",
                    "description": "the first number to add"
                },
                "b": {
                    "type": "int",
                    "description": "the second number to add"
                }
            },
            "required": ["a", "b"]
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'mul',
        'description': 'use this function to calculate the multiplication of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'the first number to multiply',
                },
                'b': {
                    'type': 'int',
                    'description': 'the second number to multiply',
                },
            },
            'required': ['a', 'b'],
        },
    }
}]
messages = []
messages.append({'role': 'system', 'content': 'if asked to do math, call function tools.'})
messages.append({'role': 'user', 'content': 'Compute (3+5)*2'})

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:45678/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func1_name = response.choices[0].message.tool_calls[0].function.name
func1_args = response.choices[0].message.tool_calls[0].function.arguments
func1_out = eval(f'{func1_name}(**{func1_args})')
print(func1_out)

messages.append({
    'role': 'assistant',
    'content': response.choices[0].message.content
})
messages.append({
    'role': 'environment',
    'content': f'3+5={func1_out}',
    'name': 'plugin'
})
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func2_name = response.choices[0].message.tool_calls[0].function.name
func2_args = response.choices[0].message.tool_calls[0].function.arguments
func2_out = eval(f'{func2_name}(**{func2_args})')
print(func2_out)
```

python internlm2_5_func.py

运行结果：
![alt text](images/image-7.png)

不同模型的函数调用能力差异明显。笔者实验中使用 1.8b 版本的书生 internLM 2.5 模型，第一次调用大多不成功，经过几轮对话后才能准确调用；而 7b 版本第一次调用就可以正确使用函数输出结果。
