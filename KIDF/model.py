# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import BRELS
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(

)

class MedicalDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}, self.labels[idx]

def get_num_classes(labels_path):
    """自动统计真实类别数"""
    labels = torch.load(labels_path, weights_only=False)
    return len(set(labels.numpy()))

def train_model(model, train_loader, dev_loader, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-6, weight_decay=0.01)

    for epoch in range(num_epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0
        correct = total = 0
        with torch.no_grad():
            for batch in dev_loader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Val Loss: {val_loss/len(dev_loader):.4f}, Val Acc: {100*correct/total:.2f}%')

if __name__ == "__main__":
    # 1. 加载新数据集
    train_inputs = torch.load("", weights_only=False)
    train_labels = torch.load("", weights_only=False)
    dev_inputs   = torch.load("",   weights_only=False)
    dev_labels   = torch.load("",   weights_only=False)

    # 2. 自动获得类别数
    num_classes = get_num_classes("")
    print(f"实际类别数: {num_classes}")

    # 3. 构建 DataLoader
    train_dataset = MedicalDataset(train_inputs, train_labels)
    dev_dataset   = MedicalDataset(dev_inputs,   dev_labels)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader    = DataLoader(dev_dataset,   batch_size=32, shuffle=False)

    # 4. 模型 & 训练
    model = BRELS(num_classes=num_classes)
    train_model(model, train_loader, dev_loader, num_epochs=)

    # 5. 保存最优权重
    torch.save(model.state_dict(), "")
    print("训练完成，权重已保存 -> ")