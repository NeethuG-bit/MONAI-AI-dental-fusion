print("STARTING SCRIPT...")
import torch
import torch.nn as nn
import torch.optim as optim

from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms
from model import DentalFusionNetwork
from monai.metrics import DiceMetric
from visualize import visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate data
pan = generate_panoramic()
cbct = generate_cbct()
soft = generate_soft_tissue()

# Transforms
pan_t, cbct_t, soft_t = get_transforms()

pan = pan_t(pan).unsqueeze(0).float().to(device)
cbct = cbct_t(cbct).unsqueeze(0).float().to(device)
soft = soft_t(soft).unsqueeze(0).float().to(device)

# Model
model = DentalFusionNetwork().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Starting training...")

for epoch in range(3):  # small epochs for now
    model.train()

    optimizer.zero_grad()

    output, _ = model(pan, cbct, soft)

    loss = criterion(output, cbct)  # using CBCT as pseudo ground truth

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete. Running visualization...")

model.eval()
with torch.no_grad():
    output, _ = model(pan, cbct, soft)

visualize(pan, cbct, soft, output)    