# RAFT长文本沐曦训练全流程

目录结构
```
AI4MED/               # 在LLaMA-Factory框架的基础上构建
├── alog              # 待处理的医疗数据集   
├── getadata.py       # 预处理
├── run.py            # 推理
├── eval.py           # 评估
├── models            # raft_share
├── utils             # raft_share
└── readme.md
```

# 数据处理

采用Qdrant对于每个文本块划分索引，然后获取top5相关文档和top5不相关文档（参考RAG-FiT做法）

提供p和num_distract两个参数，p指的是包含相关文档的比例；num_distract指的是p%的文档中，相关文档:不相关文档=1:num_distract。

先合并训练集和验证集，在训练的时候再划分（对齐LLaMA-Factory格式）

- 预处理代码：

```
python getdata.py --datasets all_train all_dev --p 0.8 --num_distract 3
```

可以通过下面这个方式查看处理后的数据集的情况
```
python read.py
```

数据的结构如下,根据RAFT论文的Figure 6，训练集在包含4个文档的时候效果基本最好；并且p=0.8，num_distract=3。经过测试，处理后最长的instruction的token数为11.32K.

```
{
    "instruction": "preamble+文本块1+文本块2+文本块3+文本块4+问题文本",
    "output": "标准答案",
    "golden": true
}
```
1. instruction：输入的prompt
2. output：选择题答案
3. golden: 数据类型标记，文本块中包含相关文档

# 训练
使用LLaMA-Factory框架，根据官方提供的sft微调示例脚本修改：

8卡MetaX C500预估训练时间为43小时，目前训练方式支持的最长token数约为15K

- 训练代码

```
mv processed_data.jsonl data
accelerate launch --config_file accelerate_config.yaml src/train.py examples/train_lora/llama3_lora_sft.yaml
```

# 推理
可以开多个终端，仅修改CUDA_VISIBLE_DEVICES指定的显卡，然后运行相同的命令，即可实现类似于数据并行的效果，双卡RTX 3090约7小时完成。

- 推理代码

```
CUDA_VISIBLE_DEVICES=0 python run.py --system reader --dataset all_test --llm_name /mnt/public/code/wangzr/yjy/Meta-Llama-3.1-8B-Instruct --plan_name system=planner_addret,dataset=all_test,debug=False
```

# 测评

- 评估代码

```
python eval.py alog/system=reader,dataset=all_test,llm_name=/home/hdd/model/Qwen2.5-7B,plan_name=system=planner_addret,dataset=all_test,debug=False,debug=False
```# AI4MED
# AI4MED
