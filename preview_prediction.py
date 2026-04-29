import os
import nibabel as nib
import matplotlib.pyplot as plt

input_path = "dental_cbct_segmentation/testing_data/test_cbct.nii.gz"
pred_path = "dental_cbct_segmentation/outputs/pred_mask.nii.gz"
output_path = "dental_cbct_segmentation/docs/sample_output.png"

os.makedirs("dental_cbct_segmentation/docs", exist_ok=True)

image = nib.load(input_path).get_fdata()
pred = nib.load(pred_path).get_fdata()

mid = image.shape[2] // 2

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].imshow(image[:, :, mid], cmap="gray")
axs[0].set_title("Input CBCT Slice")
axs[0].axis("off")

axs[1].imshow(image[:, :, mid], cmap="gray")
axs[1].imshow(pred[:, :, mid], cmap="jet", alpha=0.35)
axs[1].set_title("Predicted Mask Overlay")
axs[1].axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved preview to {output_path}")