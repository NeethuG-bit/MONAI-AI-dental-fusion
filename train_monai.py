import os
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsurechannelFirstd,
    ScaleIntensityd, ResizeD
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- PATH PLACEHOLDER --------
data_dir = "datasets/toothfairy"
data_dir = r"C:\Users\neeth\Downloads\toothfairy"

#mYou will update these after download
images = []
labels = []

data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# -------- TRANSFORMS --------
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsurechannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    ResizeD(keys=["image", "label"], spatial_size=(64, 64, 32))
])

dataset = Dataset(data=data, transform=transforms)
loader = DataLoader(dataset, batch_size=1)

# -------- MODEL --------
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64),
    strides=(2, 2),
    num_res_units=2
).to(device)

loss_fn = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------- TRAIN LOOP --------
for epoch in range(2):
    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)

        pred = model(img)
        loss = loss_fn(pred, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")

print("Training setup ready.")        