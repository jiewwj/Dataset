# evaluate.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('')
from train import (
    MedicalDataset,
    get_num_classes_and_mapper,
    BRELS
)
# --------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = acc_sum = n = 0
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in tqdm(loader, desc="Eval"):
        x, y = {k: v.to(device) for k, v in x.items()}, y.to(device)
        out = model(**x)
        loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        acc_sum += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, acc_sum / n

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_classes, _, dev_labels = get_num_classes_and_mapper(
        ""
    )
    dev_inputs = torch.load("", weights_only=False)
    dev_set = MedicalDataset(dev_inputs, dev_labels)
    dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)


    model = BRELS(num_classes=num_classes)
    model.load_state_dict(torch.load("",
                                     map_location=device))
    model.to(device)

    val_loss, val_acc = evaluate(model, dev_loader, device)
    print(f"Dev Loss: {val_loss:.4f} | Dev Acc: {val_acc:.4f}")