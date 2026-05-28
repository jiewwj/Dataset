import json, requests, pathlib
from tqdm import tqdm

IN_FILE  = pathlib.Path(r"输入JSON文件路径")
OUT_FILE = pathlib.Path(r"输出三元组JSONL文件路径")

URL   = "API接口地址"
KEY   = "API密钥"
MODEL = "模型名称"

PROMPT = """
提示词模板

"""

def chat_qwen(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 2048, "stream": False}
    r = requests.post(URL, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def extract_triples(data: str, pid: str) -> dict:
    text = chat_qwen(PROMPT.format(pid=pid, data=data))
    ents = [l.strip() for l in text.split("<实体>")[1].split("</实体>")[0].splitlines() if l.strip()] if "<实体>" in text else []
    rels = [l.strip() for l in text.split("<关系>")[1].split("</关系>")[0].splitlines() if l.strip()] if "<关系>" in text else []
    if not ents and not rels:
        ents = [f"{pid}##患者"]
    return {"entities": ents, "relations": rels}

def main():
    total = len(json.loads(IN_FILE.read_text(encoding='utf-8')))
    done  = sum(1 for _ in open(OUT_FILE, "r", encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"已抽取 {done} 条，继续剩余 {total - done} 条...")

    all_items = json.loads(IN_FILE.read_text(encoding='utf-8'))
    todo      = all_items[done:]

    for idx, item in enumerate(tqdm(todo, total=len(todo)), start=done + 1):
        pid = f"P{idx:05d}"
        triples = extract_triples(item["query"], pid)
        with open(OUT_FILE, "a", encoding="utf-8") as fout:
            fout.write(json.dumps({"pid": pid, **triples}, ensure_ascii=False) + "\n")
            if idx % 100 == 0:
                fout.flush()

    final_cnt = sum(1 for _ in open(OUT_FILE, "r", encoding="utf-8"))
    print(f"✅ 抽取完成！输出文件最终条数：{final_cnt}")

if __name__ == "__main__":
    main()