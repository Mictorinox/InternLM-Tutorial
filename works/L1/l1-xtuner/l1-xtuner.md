本文为笔者参与书生大模型实战营第四期的关卡任务实现记录。在此仅作流程演示和简要说明。如果想了解更多关于原理和不同实现方式的细节，推荐参考主办方提供的[说明文档](https://github.com/InternLM/Tutorial/blob/camp4/docs/L1/XTuner/README.md)。

# 课程任务

l1-xtuner 课程任务如下：

1. 使用 XTuner 微调 InternLM2-Chat-7B 实现自己的小助手认知模型
2. （选做）将模型上传到 HuggingFace/Modelscope/魔乐平台并部署
3. （选做）获取浦语 api 创建自己的数据用于微调

本文小标题如下

1. 微调实现
   1.1 配置环境
   1.2 准备数据
   1.3 启动微调
   1.4 权重转换
   1.5 模型合并
   1.6 模型调用
2. HuggingFace 部署微调模型


## 1.1 配置环境

### 安装 XTuner

新建 conda 环境"xtuner-env"并激活

使用源码安装 XTuner：

```shell
git clone https://github.com/InternLM/xtuner.git
cd ./xtuner
pip install  -e '.[all]'
```

之后使用`xtuner list-cfg`验证安装

### 安装依赖包：

```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.39.0
```

## 1.2 准备数据

课程材料已经准备了数据，数据是 json 格式的对话记录，在此只需要把“尖米”替换成自己的名字。

## 1.3 启动微调

### 修改 Config 文件

（config 文件包含设置）

```shell
cd ~/Arthur-L1/finetune
mkdir ./config
cd config
xtuner copy-cfg internlm2_5_chat_7b_qlora_alpaca_e3 ./
```

修改 config 文件
`"~/Tutorial/configs/internlm2_5_chat_7b_qlora_alpaca_e3_copy.py"`

```shell
cd ~/Arthur-L1/finetune
xtuner train ./config/internlm2_5_chat_7b_qlora_alpaca_e3_copy.py --deepspeed deepspeed_zero2 --work-dir ./work_dirs/assistTuner
```

微调完成输出如下：

![alt text]("https://github.com/Mictorinox/InternLM-Tutorial/tree/camp4/works/L1/l1-llamaindex/images/image.png")

## 1.4 权重转换

将使用 pytorch 训练得到的权重文件转换为通用的 huggingface 格式文件

```shell
cd ~/Arthur-L1/finetune/work_dirs/assistTuner
pth_file=`ls -t /root/Arthur-L1/finetune/work_dirs/assistTuner/*.pth | head -n 1`
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_5_chat_7b_qlora_alpaca_e3_copy.py ${pth_file} ./hf
```

权重转换完成后输出：

![alt text]("https://github.com/Mictorinox/InternLM-Tutorial/tree/camp4/works/L1/l1-llamaindex/images/image-1.png")

输出成功后，可以在./finetune 文件夹下找到 hf 格式的模型权重

![alt text]("https://github.com/Mictorinox/InternLM-Tutorial/tree/camp4/works/L1/l1-llamaindex/images/image-2.png")

## 1.5 模型合并

```shell
cd /root/Arthur-L1/finetune/work_dirs/assistTuner
conda activate xtuner-2

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert merge /root/Arthur-L1/finetune/models/internlm2_5-7b-chat ./hf ./merged --max-shard-size 2GB
```

模型合并输出：

![alt text]("https://github.com/Mictorinox/InternLM-Tutorial/tree/camp4/works/L1/l1-llamaindex/images/image-3.png")

合并后可以在./merged 文件夹下找到

![alt text]("https://github.com/Mictorinox/InternLM-Tutorial/tree/camp4/works/L1/l1-llamaindex/images/image-4.png")

streamlit run /root/Tutorial/tools/L1_XTuner_code/xtuner_streamlit_demo.py

/root/Arthur-L1/finetune/work_dirs/assistTuner/merged

## 1.6 模型调用

WebUI 对话效果：

![alt text]("https://github.com/Mictorinox/InternLM-Tutorial/tree/camp4/works/L1/l1-llamaindex/images/image-5.png")
