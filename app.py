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
    overview_cards,
    challenge_cards,
    modalities_cards,
    pipeline_cards,
    upload_placeholders,
    use_case_tabs,
    platform_cards,
    outcomes_cards,
    clinical_summary_box,
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
    page = st.radio(
        "Go to section",
        ["Overview", "Live Demo", "Use Cases", "Platform"]
    )
    st.markdown("---")
    run_demo = st.button("Run Fusion Demo")
    show_shapes = st.toggle("Show tensor shapes", value=False)
    use_colored_output = st.toggle("Colored output", value=True)
    st.markdown("---")
    st.caption("Clinical product-style proof of concept")

hero_section()

if page == "Overview":
    overview_cards()
    challenge_cards()
    modalities_cards()
    pipeline_cards()
    upload_placeholders()
    outcomes_cards()
    clinical_summary_box()

elif page == "Live Demo":
    section_title(
        "Live Fusion Demo",
        "Synthetic data-backed demo for product presentation and interface walkthrough."
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        st.markdown('<span class="mini-tag">Panoramic</span>', unsafe_allow_html=True)
        st.markdown('<span class="mini-tag">CBCT</span>', unsafe_allow_html=True)
        st.markdown('<span class="mini-tag">Soft Tissue</span>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<span class="mini-tag">MONAI</span>', unsafe_allow_html=True)
        st.markdown('<span class="mini-tag">Fusion Demo</span>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<span class="mini-tag">Clinical UI</span>', unsafe_allow_html=True)
        st.markdown('<span class="mini-tag">POC</span>', unsafe_allow_html=True)

    if run_demo:
        with st.spinner("Running multimodal inference..."):
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
                st.error(f"Runtime error while executing model inference: {e}")
                st.stop()

        st.success("Fusion inference completed successfully")

        pan_np = pan_tensor.cpu().numpy()[0, 0]
        cbct_np = cbct_tensor.cpu().numpy()[0, 0]
        soft_np = soft_tensor.cpu().numpy()[0, 0]
        output_np = output_tensor.cpu().numpy()[0, 0]

        if show_shapes:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Panoramic", str(pan_np.shape))
            m2.metric("CBCT", str(cbct_np.shape))
            m3.metric("Soft Tissue", str(soft_np.shape))
            m4.metric("Output", str(output_np.shape))

        def get_slice(img):
            if img.ndim == 2:
                return img
            if img.ndim == 3:
                return img[:, :, img.shape[2] // 2]
            return img

        cmap = "viridis" if use_colored_output else "gray"

        st.markdown("### Fusion Visualization")

        fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))

        axs[0].imshow(get_slice(pan_np), cmap=cmap)
        axs[0].set_title("Panoramic")

        axs[1].imshow(get_slice(cbct_np), cmap=cmap)
        axs[1].set_title("CBCT")

        axs[2].imshow(get_slice(soft_np), cmap=cmap)
        axs[2].set_title("Soft Tissue")

        axs[3].imshow(get_slice(output_np), cmap=cmap)
        axs[3].set_title("Fused Output")

        for ax in axs:
            ax.axis("off")

        st.pyplot(fig, use_container_width=True)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
        buffer.seek(0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="summary-card">
                    <h4>Inference Summary</h4>
                    CPU-based demo execution for cloud-friendly product presentation.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="summary-card">
                    <h4>Fusion Strategy</h4>
                    Feature-level multimodal combination across panoramic, CBCT, and soft tissue streams.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.download_button(
                "Download Demo Figure",
                data=buffer,
                file_name="dental_ai_fusion_output.png",
                mime="image/png",
            )

        clinical_summary_box()
    else:
        st.info("Use the sidebar button to run the live fusion demo.")

elif page == "Use Cases":
    use_case_tabs()
    outcomes_cards()

elif page == "Platform":
    platform_cards()
    pipeline_cards()
    clinical_summary_box()

footer_note()