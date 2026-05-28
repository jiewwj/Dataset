# train_transe.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocab():
    ents, rels = set(), set()
    for split in ['train', 'dev']:
        with open(f'数据集{split}的三元组JSONL文件路径', encoding='utf8') as f:
            for line in f:
                d = json.loads(line)
                for r in d.get('relations', []):
                    parts = r.split('##')
                    if len(parts) == 3:
                        h, r, t = parts
                        ents.update([h, t])
                        rels.add(r)
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    torch.save(ent2id, '实体到ID映射文件保存路径.pt')
    torch.save(rel2id, '关系到ID映射文件保存路径.pt')
    return ent2id, rel2id

class TripleDataset(Dataset):
    def __init__(self, fn, ent2id, rel2id):
        self.tri = []
        with open(fn, encoding='utf8') as f:
            for line in f:
                d = json.loads(line)
                for r in d.get('relations', []):
                    parts = r.split('##')
                    if len(parts) == 3:
                        h, r, t = parts
                        if h in ent2id and t in ent2id and r in rel2id:
                            self.tri.append((ent2id[h], rel2id[r], ent2id[t]))
    def __len__(self): return len(self.tri)
    def __getitem__(self, idx): return self.tri[idx]

class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100, p=2):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        self.p = p
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        h = F.normalize(self.ent(h), p=2, dim=1)
        r = F.normalize(self.rel(r), p=2, dim=1)
        t = F.normalize(self.ent(t), p=2, dim=1)
        return (h + r - t).norm(p=self.p, dim=1)

def corrupt(batch, n_ent):
    h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
    neg = torch.randint(n_ent, (h.size(0),), device=h.device)
    mask = torch.rand(h.size(0), device=h.device) > 0.5
    h2 = h.clone(); h2[mask] = neg[mask]
    t2 = t.clone(); t2[~mask] = neg[~mask]
    return torch.stack([h2, r, t2], 1)

def train():
    ent2id, rel2id = build_vocab()
    ds = TripleDataset('训练集三元组JSONL文件路径', ent2id, rel2id)
    loader = DataLoader(ds, batch_size=, shuffle=True, drop_last=True)
    model = TransE(len(ent2id), len(rel2id)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=)
    for epoch in range():
        model.train()
        total = 0.
        for batch in loader:
            batch = torch.stack(batch, 1).to(device)
            neg = corrupt(batch, len(ent2id))
            pos_score = model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_score = model(neg[:, 0], neg[:, 1], neg[:, 2])
            loss = F.relu(pos_score - neg_score + 1.0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f'Transe Epoch {epoch+1:02d}  loss={total/len(loader):.4f}')
    torch.save(model.rel.weight.data, '关系嵌入张量保存路径.pt')
    print('TransE done → 关系嵌入张量已保存')

if __name__ == '__main__':
    train()