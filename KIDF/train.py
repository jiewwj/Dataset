import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from sklearn.metrics import accuracy_score, f1_score
import json, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS =
BATCH_SIZE =
LR =
REL_DIM =

rel2id = torch.load('关系到ID的映射文件路径.pt')
rel_emb = torch.load('关系嵌入张量文件路径.pt')

def sample2relid_list(triple_dict):
    ids = []
    for r in triple_dict.get('relations', []):
        parts = r.split('##')
        if len(parts) == 3 and parts[1] in rel2id:
            ids.append(rel2id[parts[1]])
    ids = ids[:8]
    vec = torch.zeros(8, dtype=torch.long)
    vec[:len(ids)] = torch.tensor(ids, dtype=torch.long)
    return vec

class ClfDataset(Dataset):
    def __init__(self, inputs, labels, triples_file):
        self.inputs, self.labels = inputs, labels
        with open(triples_file, encoding='utf8') as f:
            self.triples = [json.loads(l) for l in f]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        rel_vec = sample2relid_list(self.triples[idx])
        item = {k: v[idx] for k, v in self.inputs.items()}
        item['rel_vec'] = rel_vec
        return item, self.labels[idx]

def load_clf(split):
    inp = torch.load(f'数据集{split}_输入文件路径.pt', weights_only=False)
    lbl = torch.load(f'数据集{split}_标签文件路径.pt', weights_only=False)
    return inp, lbl

train_inputs, train_labels = load_clf('train')
dev_inputs, dev_labels = load_clf('dev')
num_classes = len(set(train_labels))

train_loader = DataLoader(
    ClfDataset(train_inputs, train_labels, '训练集三元组JSONL文件路径'),
    batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(
    ClfDataset(dev_inputs, dev_labels, '验证集三元组JSONL文件路径'),
    batch_size=BATCH_SIZE, shuffle=False)

class LRCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim * 2, output_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        conv_out = self.conv(lstm_out.transpose(1, 2))
        pooled = self.pool(conv_out).transpose(1, 2)
        return pooled

class BRELS(nn.Module):
    def __init__(self, num_classes, rel_emb_tensor, dim=):
        super().__init__()
        self.bert = BertModel.from_pretrained('中文BERT模型路径', local_files_only=True)
        for name, p in self.bert.named_parameters():
            if 'layer.10' not in name and 'layer.11' not in name:
                p.requires_grad = False
        self.drop = nn.Dropout(0.3)
        self.rel_emb = nn.Embedding.from_pretrained(rel_emb_tensor, freeze=True)
        self.lrcnn = LRCNN(768 + dim, 256, 128)
        self.fc = nn.Linear(768 + 128, num_classes)

    def forward(self, input_ids, attention_mask, rel_vec, token_type_ids=None):
        seq_out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids).last_hidden_state
        seq_out = self.drop(seq_out)
        B, T, _ = seq_out.shape
        rel = self.rel_emb(rel_vec).mean(dim=1)
        rel = rel.unsqueeze(1).expand(-1, T, -1)
        fused = torch.cat([seq_out, rel], dim=-1)
        global_feat = self.lrcnn(fused).mean(dim=1)
        cls_feat = seq_out[:, 0]
        cat = torch.cat([cls_feat, global_feat], dim=1)
        logits = self.fc(cat)
        return logits

@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_loss, all_pred, all_true = [], [], []
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        labels = labels.to(device)
        rel_vec = inputs.pop('rel_vec').to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs, rel_vec=rel_vec)
        loss = criterion(logits, labels)
        all_loss.append(loss.item())
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(labels.cpu().numpy())
    avg_loss = sum(all_loss) / len(all_loss)
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='macro')
    return avg_loss, acc, f1

model = BRELS(num_classes=num_classes, rel_emb_tensor=rel_emb).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

best_acc = 0.
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.
    for inputs, labels in train_loader:
        labels = labels.to(device)
        rel_vec = inputs.pop('rel_vec').to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        optimizer.zero_grad()
        logits = model(**inputs, rel_vec=rel_vec)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    dev_loss, dev_acc, dev_f1 = evaluate(dev_loader)
    print(f'Epoch {epoch:02d}  train_loss={total_loss/len(train_loader):.4f}  '
          f'dev_loss={dev_loss:.4f}  acc={dev_acc:.4f}  macro-F1={dev_f1:.4f}')
    if dev_acc > best_acc:
        best_acc = dev_acc
        torch.save(model.state_dict(), '最佳模型权重保存路径.pth')
        print('  ★ best model saved')
torch.save(model.state_dict(), '最后模型权重保存路径.pth')
print('Training finished! best_acc =', best_acc)