import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_manifest import load_manifest, split_by_source
from models import ResNet50_6ch
from mmf import eps_feature_stack

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT = r".\runs_resnet"
os.makedirs(OUT, exist_ok=True)

def read_rgb01(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)

class Tiles6ChClsDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        label = 0.0 if self.df.loc[idx, "label"] == "healthy" else 1.0
        rgb = read_rgb01(p)
        eps3 = eps_feature_stack(rgb)
        x6 = np.concatenate([rgb, eps3], axis=2)
        x6 = torch.from_numpy(x6).permute(2,0,1).float()
        y = torch.tensor(label).float()
        return x6, y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = load_manifest(MANIFEST)
    tr_df, va_df = split_by_source(df, test_size=0.2, seed=42)

    train_ds = Tiles6ChClsDataset(tr_df)
    val_ds   = Tiles6ChClsDataset(va_df)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    model = ResNet50_6ch(in_ch=6).to(device)

    # reduce overfit: start with small LR
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = 1e9
    for epoch in range(1, 9):
        model.train()
        tr = 0.0
        for x, y in tqdm(train_loader, desc=f"Train ResNet {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= max(1, len(train_loader))

        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                va += loss_fn(logits, y).item()
        va /= max(1, len(val_loader))

        print(f"Epoch {epoch}: train={tr:.5f} val={va:.5f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(OUT, "resnet_6ch_best.pt"))
            print("Saved best ResNet 6ch.")

    print("Done. Best val:", best)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
