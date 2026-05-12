import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from monai.deploy.core import Application
from monai.networks.nets import UNet


class DentalFusionApp(Application):
    def __init__(self):
        self.input_path = "dental_cbct_segmentation/testing_data/test_cbct.nii.gz"
        self.model_path = "dental_cbct_segmentation/models/model.pt"
        self.output_dir = "monai_deploy_outputs"
        self.output_path = os.path.join(self.output_dir, "deploy_pred_mask.nii.gz")
        self.preview_path = os.path.join(self.output_dir, "deploy_preview.png")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__()

    def load_input(self):
        print("📥 Loading CBCT input...")
        self.image_nii = nib.load(self.input_path)
        self.image = self.image_nii.get_fdata().astype(np.float32)
        print("Input shape:", self.image.shape)

    def preprocess(self):
        print("⚙️ Preprocessing CBCT volume...")
        image = self.image

        image_min = image.min()
        image_max = image.max()

        if image_max - image_min > 1e-6:
            image = (image - image_min) / (image_max - image_min)

        self.image_tensor = (
            torch.tensor(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

    def run_inference(self):
        print("🧠 Running MONAI segmentation model...")

        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32),
            strides=(2, 2),
            num_res_units=1,
        ).to(self.device)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        with torch.no_grad():
            pred = model(self.image_tensor)
            pred_mask = (torch.sigmoid(pred) > 0.5).float()

        self.pred_mask = pred_mask.cpu().numpy()[0, 0].astype(np.float32)
        print("Prediction shape:", self.pred_mask.shape)

    def save_output(self):
        print("💾 Saving NIfTI segmentation output...")
        os.makedirs(self.output_dir, exist_ok=True)

        out_nii = nib.Nifti1Image(self.pred_mask, self.image_nii.affine)
        nib.save(out_nii, self.output_path)

        print("Saved output:", self.output_path)

    def save_preview(self):
        print("🖼️ Generating deployment preview...")

        mid = self.image.shape[2] // 2

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(self.image[:, :, mid], cmap="gray")
        plt.title("Input CBCT")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(self.image[:, :, mid], cmap="gray")
        plt.imshow(self.pred_mask[:, :, mid], cmap="jet", alpha=0.35)
        plt.title("Segmentation Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(self.preview_path, dpi=200)
        plt.close()

        print("Saved preview:", self.preview_path)

    def compose(self):
        print("====================================")
        print("Dental AI MONAI Deploy Pipeline")
        print("====================================")

        self.load_input()
        self.preprocess()
        self.run_inference()
        self.save_output()
        self.save_preview()

        print("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    app = DentalFusionApp()