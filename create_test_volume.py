import numpy as np
import nibabel as nib
import os

os.makedirs("dental_cbct_segmentation/testing_data", exist_ok=True)

volume = np.random.rand(64, 64, 32).astype(np.float32)

affine = np.eye(4)
img = nib.Nifti1Image(volume, affine)

nib.save(img, "dental_cbct_segmentation/testing_data/test_cbct.nii.gz")

print("Created test CBCT volume successfully.")