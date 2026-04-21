import io
import numpy as np
from PIL import Image
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import DentalFusionNetwork
from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms

from styles import load_css
from demo_sections import (
    hero_section,
    kpi_row,
    workflow_banner,
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
    st.caption("Client-facing clinical product demo")

hero_section()
kpi_row()
workflow_banner()

def image_to_gray_array(uploaded_file, target_size=(64, 64)):
    img = Image.open(uploaded_file).convert("L").resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = (arr / 255.0) * 2000 - 1000
    return arr

if page == "Overview":
    overview_cards()
    challenge_cards()
    modalities_cards()
    pipeline_cards()
    outcomes_cards()
    clinical_summary_box()

elif page == "Live Demo":
    section_title(
        "Live Fusion Dashboard",
        "Preview uploads for a product-like experience, then run the current fusion demo."
    )

    pan_file, cbct_file, soft_file = upload_placeholders()

    preview_cols = st.columns(3)

    with preview_cols[0]:
        st.markdown('<div class="preview-card"><strong>Panoramic Preview</strong></div>', unsafe_allow_html=True)
        if pan_file:
            st.image(pan_file, use_container_width=True)
        else:
            st.info("No panoramic upload provided")

    with preview_cols[1]:
        st.markdown('<div class="preview-card"><strong>CBCT Preview</strong></div>', unsafe_allow_html=True)
        if cbct_file:
            st.image(cbct_file, use_container_width=True)
        else:
            st.info("No CBCT preview provided")

    with preview_cols[2]:
        st.markdown('<div class="preview-card"><strong>Soft Tissue Preview</strong></div>', unsafe_allow_html=True)
        if soft_file:
            st.image(soft_file, use_container_width=True)
        else:
            st.info("No soft tissue upload provided")

    st.markdown("---")

    if run_demo:
        with st.spinner("Running multimodal fusion inference..."):
            device = torch.device("cpu")

            # Use uploads if available, otherwise synthetic fallback
            if pan_file:
                pan = image_to_gray_array(pan_file, target_size=(64, 64))
            else:
                pan = generate_panoramic((64, 64))

            if cbct_file:
                cbct_2d = image_to_gray_array(cbct_file, target_size=(64, 64))
                cbct = np.repeat(cbct_2d[:, :, None], 32, axis=2)
            else:
                cbct = generate_cbct((64, 64, 32))

            if soft_file:
                soft_img = Image.open(soft_file).convert("L").resize((64, 64))
                soft = np.array(soft_img).astype(np.float32)
                soft = (soft / 255.0) * 15.0
            else:
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

        section_title("Fusion Visualization", "Synthetic or uploaded-preview-driven demonstration output")

        result_cols = st.columns(4)
        visuals = [
            ("Panoramic", get_slice(pan_np)),
            ("CBCT", get_slice(cbct_np)),
            ("Soft Tissue", get_slice(soft_np)),
            ("Fused Output", get_slice(output_np)),
        ]

        for col, (title, image) in zip(result_cols, visuals):
            with col:
                st.markdown(f'<div class="preview-card"><strong>{title}</strong></div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                ax.imshow(image, cmap=cmap)
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)

        fig_full, axs = plt.subplots(1, 4, figsize=(15, 4.2))
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

        buffer = io.BytesIO()
        fig_full.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
        buffer.seek(0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="summary-card">
                    <h4>Inference Summary</h4>
                    CPU-based cloud demo optimized for client-facing presentation and product walkthrough.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="summary-card">
                    <h4>Clinical Interpretation</h4>
                    Use the fused output as a decision-support style visualization rather than a final diagnostic conclusion.
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
        st.info("Upload optional preview images, then use the sidebar button to run the live fusion demo.")

elif page == "Use Cases":
    use_case_tabs()
    outcomes_cards()

elif page == "Platform":
    platform_cards()
    pipeline_cards()
    clinical_summary_box()

footer_note()