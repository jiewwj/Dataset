import torch
from torch.utils.data import Dataset, DataLoader
from model import BRELS
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MedicalDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}, self.labels[idx]

@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in test_loader:
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        logits = model(**inputs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f'Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}')

if __name__ == '__main__':
    test_inputs = torch.load(r"测试集输入文件路径.pt")
    test_labels = torch.load(r"测试集标签文件路径.pt")

    test_dataset = MedicalDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BRELS(num_classes=2).to(device)
    model.load_state_dict(torch.load(r"模型权重文件路径.pth", map_location=device))

    evaluate_model(model, test_loader)