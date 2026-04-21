import io
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import DentalFusionNetwork
from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms

from styles import load_css
from demo_sections import (
    hero_section,
    problem_section,
    modalities_section,
    pipeline_section,
    use_cases_section,
    platform_section,
    outcomes_section,
    footer_note,
    section_title,
)

st.set_page_config(
    page_title="Dental AI Fusion Platform",
    page_icon="🦷",
    layout="wide",
)

st.markdown(load_css(), unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    st.markdown('<span class="tag">MONAI</span><span class="tag">Fusion</span><span class="tag">Clinical Demo</span>', unsafe_allow_html=True)
    st.markdown("---")
    run_demo = st.button("Run Live Fusion Demo")
    show_shapes = st.toggle("Show tensor shapes", value=False)
    st.markdown("---")
    st.caption("Proof-of-concept multimodal dental imaging demo")

hero_section()
problem_section()
modalities_section()
pipeline_section()

section_title("Live Fusion Demo")

demo_placeholder = st.container()

if run_demo:
    with demo_placeholder:
        with st.spinner("Running multimodal fusion inference..."):
            device = torch.device("cpu")

            pan = generate_panoramic((64, 64))
            cbct = generate_cbct((64, 64, 32))
            soft = generate_soft_tissue((64, 64))

            pan_t, cbct_t, soft_t = get_transforms()

            pan_tensor = pan_t(pan).unsqueeze(0).float().to(device)
            cbct_tensor = cbct_t(cbct).unsqueeze(0).float().to(device)
            soft_tensor = soft_t(soft).unsqueeze(0).float().to(device)

            model = DentalFusionNetwork().to(device)
            model.eval()

            try:
                with torch.no_grad():
                    output_tensor, features = model(pan_tensor, cbct_tensor, soft_tensor)
            except Exception as e:
                st.error(f"Runtime error while executing the fusion model: {e}")
                st.stop()

        st.success("Fusion inference completed successfully")

        pan_np = pan_tensor.cpu().numpy()[0, 0]
        cbct_np = cbct_tensor.cpu().numpy()[0, 0]
        soft_np = soft_tensor.cpu().numpy()[0, 0]
        output_np = output_tensor.cpu().numpy()[0, 0]

        if show_shapes:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Panoramic Shape", str(pan_np.shape))
            c2.metric("CBCT Shape", str(cbct_np.shape))
            c3.metric("Soft Tissue Shape", str(soft_np.shape))
            c4.metric("Output Shape", str(output_np.shape))

        def get_slice(img):
            if img.ndim == 2:
                return img
            if img.ndim == 3:
                return img[:, :, img.shape[2] // 2]
            return img

        fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))

        axs[0].imshow(get_slice(pan_np), cmap="viridis")
        axs[0].set_title("Panoramic")

        axs[1].imshow(get_slice(cbct_np), cmap="viridis")
        axs[1].set_title("CBCT")

        axs[2].imshow(get_slice(soft_np), cmap="viridis")
        axs[2].set_title("Soft Tissue")

        axs[3].imshow(get_slice(output_np), cmap="viridis")
        axs[3].set_title("Fused Output")

        for ax in axs:
            ax.axis("off")

        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        buf.seek(0)

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.metric("Inference Mode", "CPU Demo")
        with col_b:
            st.metric("Fusion Type", "Feature-Level")
        with col_c:
            st.download_button(
                "Download Demo Output",
                data=buf,
                file_name="dental_ai_fusion_demo.png",
                mime="image/png",
            )
else:
    with demo_placeholder:
        st.info("Use the sidebar button to run the live fusion demo.")

use_cases_section()
platform_section()
outcomes_section()
footer_note()