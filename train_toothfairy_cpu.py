import glob
import os
import torch

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ResizeD,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss

device = torch.device("cpu")

data_dir = r"C:\Users\neeth\Desktop\monai_dental_project\datasets\toothfairy"

images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii*")))
labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii*")))

print("Images found:", len(images))
print("Labels found:", len(labels))

# CPU-safe tiny test
images = images[:2]
labels = labels[:2]

data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    ResizeD(keys=["image", "label"], spatial_size=(64, 64, 32)),
])

dataset = Dataset(data=data, transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(8, 16, 32),
    strides=(2, 2),
    num_res_units=1,
).to(device)

loss_fn = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1):
    model.train()
    epoch_loss = 0

    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)

        # Convert labels to biary foreground mask
        lbl = (lbl > 0).float()

        pred = model(img)
        loss = loss_fn(pred, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(loader):.4f}")

os.makedirs("dental_cbct_segmentation/models", exist_ok=True)
torch.save(model.state_dict(), "dental_cbct_segmentation/models/model.pt")

print("Training test completed.")
print("Saved model to dental_cbct_segmentation/models/model.pt")
