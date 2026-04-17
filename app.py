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

    mid = pan.shape[-1] // 2

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    axs[0].imshow(pan, cmap="viridis")
    axs[0].set_title("Panoramic")

    axs[1].imshow(cbct[:, :, mid], cmap="viridis")
    axs[1].set_title("CBCT")

    axs[2].imshow(soft[:, :, mid], cmap="viridis")
    axs[2].set_title("Soft Tissue")

    axs[3].imshow(output[:, :, mid], cmap="viridis")
    axs[3].set_title("Output")

    for ax in axs:
        ax.axis("off")

    st.pyplot(fig)