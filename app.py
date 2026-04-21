import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import DentalFusionNetwork
from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms

st.title("🦷 Dental AI Fusion Demo")

st.write("Click below to run the model")

if st.button("Run Fusion"):

    st.write("Running model...")

    device = torch.device("cpu")

    # 🔽 REDUCED SIZE (VERY IMPORTANT)
    pan = generate_panoramic((64, 64))
    cbct = generate_cbct((64, 64, 32))
    soft = generate_soft_tissue((64, 64))

    pan_t, cbct_t, soft_t = get_transforms()

    pan = pan_t(pan).unsqueeze(0).float().to(device)
    cbct = cbct_t(cbct).unsqueeze(0).float().to(device)
    soft = soft_t(soft).unsqueeze(0).float().to(device)

    model = DentalFusionNetwork().to(device)
    model.eval()

    try:
        with torch.no_grad():
            output, _ = model(pan, cbct, soft)

        st.success("Model ran successfully ✅")

    except Exception as e:
        st.error(f"Error running model: {e}")
        st.stop()

    pan = pan.cpu().numpy()[0, 0]
    cbct = cbct.cpu().numpy()[0, 0]
    soft = soft.cpu().numpy()[0, 0]
    output = output.cpu().numpy()[0, 0]

    st.write("Pan shape:", pan.shape)
    st.write("CBCT shape:", cbct.shape)
    st.write("Soft shape:", soft.shape)
    st.write("Output shape:", output.shape)

    st.write("Panoramic:", pan)
    st.write("CBCT:", cbct)
    st.write("Soft:", soft)
    st.write("Output:", output)

    st.subheader("Model Results")

    def get_slice(img):
        if img.ndim == 2:
            return img
        elif img.ndim == 3:
            return img[:, :, img.shape[2] // 2]  # slice along depth
        return img

fig, axs = plt.subplots(1, 4, figsize=(14, 4))

axs[0].imshow(get_slice(pan), cmap="viridis")
axs[0].set_title("Panoramic")

axs[1].imshow(get_slice(cbct), cmap="viridis")
axs[1].set_title("CBCT")

axs[2].imshow(get_slice(soft), cmap="viridis")
axs[2].set_title("Soft Tissue")

axs[3].imshow(get_slice(output), cmap="viridis")
axs[3].set_title("AI Output")

for ax in axs:
    ax.axis("off")

st.pyplot(fig)
st.success("Visualization generated successfully")