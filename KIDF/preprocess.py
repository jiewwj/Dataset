# preprocess.py  —— 多分类版
import json
import jieba
import torch
from transformers import BertTokenizer
from collections import Counter
import os

# ---------- 1. 参数区 ----------
TRAIN_JSON = r"训练集JSON文件路径"
DEV_JSON   = r"验证集JSON文件路径"
TEST_JSON  = r"测试集JSON文件路径"
OUT_DIR    = r"输出目录路径"
BERT_PATH  = r"本地BERT模型路径"
# --------------------------------

tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)

# ---------- 2. 工具函数 ----------
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    queries = [item['query'] for item in data]
    labels  = [item['label'] for item in data]
    return queries, labels

def build_label2id(all_labels):
    unique = sorted(set(all_labels))
    return {l: i for i, l in enumerate(unique)}

def preprocess_data(queries):
    seg = [' '.join(jieba.cut(q)) for q in queries]
    return tokenizer(seg, return_tensors='pt', padding=True, truncation=True, max_length=512)

def save_pt(inputs, labels, save_prefix, label2id):
    label_ids = [label2id[l] for l in labels]
    torch.save(inputs,  f"{save_prefix}_inputs.pt")
    torch.save(label_ids, f"{save_prefix}_labels.pt")

# ---------- 3. 主流程 ----------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    train_q, train_l = load_data(TRAIN_JSON)
    dev_q,   dev_l   = load_data(DEV_JSON)
    test_q,  test_l  = load_data(TEST_JSON)

    label2id = build_label2id(train_l + dev_l + test_l)
    print(f"[INFO] 共 {len(label2id)} 类，映射已构建")

    train_i = preprocess_data(train_q)
    dev_i   = preprocess_data(dev_q)
    test_i  = preprocess_data(test_q)

    save_pt(train_i, train_l, os.path.join(OUT_DIR, "train"), label2id)
    save_pt(dev_i,   dev_l,   os.path.join(OUT_DIR, "dev"),   label2id)
    save_pt(test_i,  test_l,  os.path.join(OUT_DIR, "test"),  label2id)

    with open(os.path.join(OUT_DIR, "label2id.json"), 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print("[OK] 全部预处理完成，文件已输出至：", OUT_DIR)