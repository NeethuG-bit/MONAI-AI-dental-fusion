import os
import torch
import nibabel as nib
import numpy as np

from monai.networks.nets import UNet
from monai.metrics import DiceMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_path = "dental_cbct_segmentation/testing_data/test_cbct.nii.gz"
pred_path = "dental_cbct_segmentation/outputs/pred_mask.nii.gz"
model_path = "dental_cbct_segmentation/models/model.pt"

image = nib.load(input_path).get_fdata().astype(np.float32)
pred = nib.load(pred_path).get_fdata().astype(np.float32)

# Since we do not have real ground truth for test_cbct,
# create s pseudo-label only for sanity evaluation.
pseudo_label = (image > image.mean()).astype(np.float32)

pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
label_tensor = torch.tensor(pseudo_label).unsqueeze(0).unsqueeze(0)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric(y_pred=pred_tensor, y=label_tensor)

dice_score = dice_metric.aggregate().item()
dice_metric.reset()

os.makedirs("dental_cbct_segmentation/evaluation", exist_ok=True)

with open("dental_cbct_segmentation/evaluation/metrics.txt", "w") as f:
    f.write("Dental CBCT Segmentation Evaluation\n")
    f.write("----------------------------------\n")
    f.write(f"Dice Score (pseudo-label sanity check): {dice_score:.4f}\n")
    f.write("\n")
    f.write("Note: This metric uses a pseudo-label generated from the test volume intensity.\n")
    f.write("It is not a clinical validation score.\n")

print(f"Dice Score: {dice_score:.4f}")
print("Saved metrics to dental_cbct_segmentation/evaluation/metrics.txt")