import nibabel as nib
import matplotlib.pyplot as plt 
import os

input_path = "dental_cbct_segmentation/testing_data/test_cbct.nii.gz"
pred_path = "monai_deploy_outputs/deploy_pred_mask.nii.gz"
output_path = "monai_deploy_outputs/deploy_preview.png"

image = nib.load(input_path).get_fdata()
pred = nib.load(pred_path).get_fdata()

mid = image.shape[2] // 2

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, mid], cmap="gray")
plt.title("Input CBCT")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image[:, :, mid], cmap="gray")
plt.imshow(pred[:, :, mid], cmap="jet", alpha=0.35)
plt.title("Deploy Segmentation Output")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=200)
print("Saved preview:", output_path)