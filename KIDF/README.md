
论文标题：An Iterative Knowledge Extraction and Dual-Channel Fusion Framework for Clinical Differential Diagnosis

模型名称：KIDF (Knowledge-guided Iterative Dual-channel Fusion)



本项目是上述论文的代码实现。核心思路是：利用TransE模型从中医知识图谱中学习关系嵌入向量，通过双通道融合机制将知识信息注入BERT分类器，提升模型对中医概念间语义关系的理解能力，用于临床鉴别诊断任务。

## 系统架构

整体数据流：

原始JSON数据 → 数据预处理 → BERT输入格式(.pt)

中医病历文本 → LLM抽取三元组 → TransE训练 → 关系嵌入向量(.pt)

最终分类：BERT编码 + 关系嵌入 → 双通道融合 → 分类结果

## 文件结构

项目根目录/
├── preprocess.py          # 数据预处理：分词、BERT编码
├── train_transe.py        # 训练TransE知识图谱嵌入模型
├── model.py               # KIDF模型定义
├── train.py               # 训练分类模型
├── evaluate.py            # 评估模型性能
└── README.md              # 项目说明文档

## 环境依赖

torch >= 1.9.0
transformers >= 4.0.0
scikit-learn >= 0.24.0
jieba >= 0.42.1
tqdm >= 4.62.0
requests >= 2.25.0

## 快速开始

步骤1：安装依赖

pip install torch transformers scikit-learn jieba tqdm requests

步骤2：准备BERT模型

下载中文BERT模型到本地目录，例如：
/path/to/bert-base-chinese/
├── config.json
├── pytorch_model.bin
└── vocab.txt

步骤3：配置文件路径

修改各脚本中的路径配置：
- preprocess.py：TRAIN_JSON、DEV_JSON、TEST_JSON、OUT_DIR、BERT_PATH
- train_transe.py：三元组JSONL文件路径
- train.py：输入数据路径、输出模型路径
- evaluate.py：测试数据路径、模型权重路径

步骤4：准备三元组数据

使用LLM从病历文本中抽取三元组，生成train__triples.jsonl和dev__triples.jsonl

步骤5：运行预处理

python preprocess.py

步骤6：训练知识图谱嵌入

python train_transe.py

步骤7：训练分类模型

python train.py

步骤8：评估模型

python evaluate.py
