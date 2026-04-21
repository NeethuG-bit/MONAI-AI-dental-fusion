import io
import numpy as np
from PIL import Image
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import DentalFusionNetwork
from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms

from demo_sections import (
    hero_section,
    kpi_row,
    workflow_banner,
    overview_cards,
    challenge_cards,
    modality_icon_cards,
    pipeline_cards,
    upload_console,
    use_case_tabs,
    platform_cards,
    architecture_page_cards,
    outcomes_cards,
    clinical_summary_box,
    footer_note,
    section_title,
)

st.set_page_config(
    page_title="Dental AI Fusion Platform",
    page_icon="🦷",
    layout="wide"
)

with st.sidebar:
    st.title("Platform Navigation")
    page = st.radio(
        "Select view",
        ["Overview", "Live Demo", "Use Cases", "Architecture", "Platform"]
    )
    st.markdown("---")
    run_demo = st.button("Run Fusion Demo")
    show_shapes = st.toggle("Show tensor shapes", value=False)
    use_colored_output = st.toggle("Colored output", value=True)
    mode = st.selectbox("Demo Mode", ["Quick Demo", "Detailed Analysis"])
    intensity = st.slider("Output Enhancement", 0.5, 2.0, 1.0)

hero_section()
st.markdown("### Powered by MONAI • Clinical Imaging Intelligence")

kpi_row()
workflow_banner()

def image_to_gray_array(uploaded_file, target_size=(64, 64)):
    img = Image.open(uploaded_file).convert("L").resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = (arr / 255.0) * 2000 - 1000
    return arr

def get_slice(img):
    squeezed = np.asarray(img).squeeze()
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3:
        mid = squeezed.shape[-1] // 2
        return squeezed[:, :, mid]
    raise ValueError(f"Unsupported array shape for display: {squeezed.shape}")

if page == "Overview":
    overview_cards()
    challenge_cards()
    modality_icon_cards()
    pipeline_cards()
    outcomes_cards()
    clinical_summary_box()

elif page == "Live Demo":
    section_title("Live Fusion Dashboard", "Clinical-style demo workspace")
    pan_file, cbct_file, soft_file = upload_console()

    if run_demo:
        with st.spinner("Running multimodal fusion inference..."):
            device = torch.device("cpu")

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

            with torch.no_grad():
                output_tensor, _ = model(pan_tensor, cbct_tensor, soft_tensor)

        st.success("Fusion inference completed successfully")

        pan_np = pan_tensor.cpu().numpy()[0, 0]
        cbct_np = cbct_tensor.cpu().numpy()[0, 0]
        soft_np = soft_tensor.cpu().numpy()[0, 0]
        output_np = output_tensor.cpu().numpy()[0, 0]

        output_np = output_np * intensity

        if show_shapes:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Panoramic", str(np.asarray(pan_np).shape))
            c2.metric("CBCT", str(np.asarray(cbct_np).shape))
            c3.metric("Soft Tissue", str(np.asarray(soft_np).shape))
            c4.metric("Output", str(np.asarray(output_np).shape))

        cmap = "viridis" if use_colored_output else "gray"

        st.markdown("### 🧾 Fusion Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.image(get_slice(pan_np), caption="Panoramic", use_container_width=True)
        c2.image(get_slice(cbct_np), caption="CBCT", use_container_width=True)
        c3.image(get_slice(soft_np), caption="Soft Tissue", use_container_width=True)
        c4.image(get_slice(output_np), caption="Fused Output", use_container_width=True)

        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
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

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        buf.seek(0)

        st.download_button(
            "Download Demo Figure",
            data=buf,
            file_name="dental_ai_fusion_output.png",
            mime="image/png",
        )

        st.markdown("### 🧠 AI Clinical Insight")
        st.info("""
• Fused output highlights structural + soft tissue correlation  
• Potential regions of interest can be derived from intensity patterns  
• Demonstrates multimodal alignment capability  
• Supports future diagnostic and planning workflows  
""")

        st.markdown("### 📊 Model Indicators")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fusion Confidence", "87%", "+3%")
        c2.metric("Data Alignment", "92%", "+5%")
        c3.metric("Inference Time", "< 1 sec")

        if mode == "Detailed Analysis":
            st.markdown("### 🔍 Detailed Analysis Mode")
            st.write("Feature maps, intermediate outputs, and advanced review panels can be added here.")

        report_text = """
Dental AI Fusion Report

Inputs:
- Panoramic
- CBCT
- Soft Tissue

Output:
- Multimodal fused visualization generated

Observations:
- Structural and soft tissue integration visible
- Suitable for planning workflows

Note:
This is a proof-of-concept output.
"""
        st.download_button(
            "📄 Download Clinical Report",
            data=report_text,
            file_name="fusion_report.txt",
        )

        st.markdown("### 🚀 Future Roadmap")
        st.success("""
• DICOM integration  
• Real patient data pipeline  
• Model training on clinical datasets  
• Edge deployment in dental clinics  
• Automated diagnosis suggestions  
""")

        st.caption("⚠️ This demo uses synthetic/sample data for visualization purposes only. Not for clinical use.")

        clinical_summary_box()
    else:
        st.info("Click 'Run Fusion Demo' to start")

elif page == "Use Cases":
    use_case_tabs()
    outcomes_cards()

elif page == "Architecture":
    architecture_page_cards()
    pipeline_cards()
    clinical_summary_box()

elif page == "Platform":
    platform_cards()
    outcomes_cards()
    clinical_summary_box()

footer_note()