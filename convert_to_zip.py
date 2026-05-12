import os
import glob
import zipfile
import nibabel as nib
import numpy as np
from PIL import Image

data_dir = r"C:\Users\neeth\Desktop\monai_dental_project\datasets\toothfairy"
images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii*")))

input_file = images[0]
output_zip = "cbct_slices_demo.zip"

os.makedirs("temp_slices", exist_ok=True)

img = nib.load(input_file).get_fdata()

for i in range(img.shape[2]):
    slice_img = img[:, :, i]
    norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-6)
    norm = (norm * 255).astype("uint8")
    Image.fromarray(norm).save(f"temp_slices/slice_{i:03d}.png")

with zipfile.ZipFile(output_zip, "w") as z:
    for f in sorted(os.listdir("temp_slices")):
        z.write(os.path.join("temp_slices", f), f)

print("ZIP created:", output_zip)
print("Source file:", input_file)            