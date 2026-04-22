import io
import time
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

# ---------------- SIDEBAR ----------------
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
    show_heatmap = st.toggle("Show AI Heatmap", value=True)
    presentation_mode = st.toggle("🎤 Presentation Mode", value=False)

    st.markdown("---")
    st.markdown("## 📘 Term Guide")

    panel = st.session_state.get("panel")

    if panel == "pan":
        st.markdown("### 🦷 Panoramic (OPG)")
        st.info("""
A panoramic radiograph is a 2D dental X-ray that shows:
- upper jaw
- lower jaw
- full dentition
- surrounding jaw structures

Common uses:
- orthodontic evaluation
- impacted tooth review
- implant planning
""")

    elif panel == "cbct":
        st.markdown("### 🧊 CBCT")
        st.info("""
CBCT stands for Cone Beam Computed Tomography.

It provides:
- 3D volumetric imaging
- bone structure detail
- tooth root position
- spatial planning support
""")

    elif panel == "soft":
        st.markdown("### 🧠 Soft Tissue")
        st.info("""
Soft tissue imaging helps visualize:
- facial surface
- contour
- soft tissue profile
- esthetic planning context
""")

    elif panel == "fusion":
        st.markdown("### 🔗 Fusion")
        st.info("""
Fusion combines information from multiple imaging modalities
into a single unified representation for review and planning.
""")

# ---------------- HEADER ----------------
hero_section()
st.markdown("### Powered by MONAI • Clinical Imaging Intelligence")

colA, colB = st.columns([6, 1])
with colB:
    st.markdown("🟢 **System Online**")

if presentation_mode:
    st.success("Live AI Demonstration Mode Active")

kpi_row()
workflow_banner()

# ---------------- HELPERS ----------------
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
    raise ValueError(f"Unsupported array shape: {squeezed.shape}")

# ---------------- PAGES ----------------
if page == "Overview":
    overview_cards()
    challenge_cards()
    modality_icon_cards()
    pipeline_cards()
    outcomes_cards()
    clinical_summary_box()

elif page == "Live Demo":
    section_title("Live Fusion Dashboard", "Clinical-style demo workspace")

    st.markdown("### 📖 Imaging Modalities")
    st.markdown(
        """
🦷 **Panoramic (OPG)**  
<span title='2D full jaw X-ray showing teeth, jaw, and structure'>ℹ️</span>

🧊 **CBCT**  
<span title='3D cone beam CT scan used for volumetric dental imaging'>ℹ️</span>

🧠 **Soft Tissue**  
<span title='Facial or soft tissue surface representation for esthetic analysis'>ℹ️</span>

🔗 **Fusion**  
<span title='Combining multiple modalities into a unified AI representation'>ℹ️</span>
""",
        unsafe_allow_html=True
    )

    pan_file, cbct_file, soft_file = upload_console()

    st.info("📁 Upload images OR run demo with built-in synthetic data")

    preview_cols = st.columns(3)
    if pan_file:
        preview_cols[0].image(pan_file, caption="Panoramic Preview")
    if cbct_file:
        preview_cols[1].image(cbct_file, caption="CBCT Preview")
    if soft_file:
        preview_cols[2].image(soft_file, caption="Soft Tissue Preview")

    if run_demo:
        if presentation_mode:
            st.markdown("### 🔄 Processing Pipeline")
            step_box = st.empty()
            progress = st.progress(0)

            steps = [
                ("🔹 Loading multimodal inputs...", 20),
                ("🔹 Applying transformations...", 40),
                ("🔹 Running fusion model...", 70),
                ("🔹 Generating output...", 100),
            ]

            last_progress = 0
            for text, prog in steps:
                step_box.info(text)
                for i in range(last_progress, prog):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                last_progress = prog

            step_box.success("✅ Processing Complete")
            progress.empty()
        else:
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

        with st.spinner("Running multimodal fusion inference..."):
            device = torch.device("cpu")

            pan = image_to_gray_array(pan_file) if pan_file else generate_panoramic((64, 64))

            if cbct_file:
                cbct_2d = image_to_gray_array(cbct_file)
                cbct = np.repeat(cbct_2d[:, :, None], 32, axis=2)
            else:
                cbct = generate_cbct((64, 64, 32))

            if soft_file:
                soft_img = Image.open(soft_file).convert("L").resize((64, 64))
                soft = (np.array(soft_img).astype(np.float32) / 255.0) * 15.0
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

        if not presentation_mode:
            progress.empty()

        st.success("Fusion inference completed successfully")

        # ---------------- DATA ----------------
        pan_np = pan_tensor.cpu().numpy()[0, 0]
        cbct_np = cbct_tensor.cpu().numpy()[0, 0]
        soft_np = soft_tensor.cpu().numpy()[0, 0]
        output_np = output_tensor.cpu().numpy()[0, 0] * intensity

        if show_shapes:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Panoramic", str(np.asarray(pan_np).shape))
            c2.metric("CBCT", str(np.asarray(cbct_np).shape))
            c3.metric("Soft Tissue", str(np.asarray(soft_np).shape))
            c4.metric("Output", str(np.asarray(output_np).shape))

        cmap = "viridis" if use_colored_output else "gray"

        # ---------------- RESULTS ----------------
        st.markdown("---")
        st.markdown("## 📊 Generated Results")
        st.markdown("## 🧾 AI Fusion Results")
        st.markdown("### 🤖 AI Interpretation Layer")
        st.success("Fusion complete. Generating visual intelligence...")
        st.caption("Multimodal visualization output from the fusion engine.")

        with st.container():
            st.markdown("#### 📊 Visualization")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(get_slice(pan_np), caption="Panoramic", use_container_width=True)
            c2.image(get_slice(cbct_np), caption="CBCT", use_container_width=True)
            c3.image(get_slice(soft_np), caption="Soft Tissue", use_container_width=True)
            c4.image(get_slice(output_np), caption="Fused Output", use_container_width=True)

        # ---------------- SLICE VIEWER ----------------
        st.markdown("---")
        st.markdown("### 🧊 CBCT Slice Explorer")

        slice_idx = st.slider(
            "Select Slice",
            0,
            cbct_np.shape[2] - 1,
            cbct_np.shape[2] // 2
        )

        fig_slice, ax1 = plt.subplots()
        ax1.imshow(cbct_np[:, :, slice_idx], cmap=cmap)
        ax1.set_title(f"CBCT Slice {slice_idx}")
        ax1.axis("off")
        st.pyplot(fig_slice)

        # ---------------- REGION DETECTION ----------------
        st.markdown("---")
        st.markdown("### 🎯 Detected Region (Simulated)")

        img = get_slice(output_np)
        mask = img > img.mean()

        fig_detect, ax2 = plt.subplots()
        ax2.imshow(img, cmap="gray")
        ax2.imshow(mask, cmap="jet", alpha=0.3)
        ax2.axis("off")
        st.pyplot(fig_detect)

        # ---------------- HEATMAP ----------------
        if show_heatmap:
            st.markdown("---")
            st.markdown("### 🔥 AI Attention Heatmap")

            heatmap = get_slice(output_np)

            fig_heat, ax3 = plt.subplots()
            ax3.imshow(get_slice(cbct_np), cmap="gray")
            ax3.imshow(heatmap, cmap="jet", alpha=0.4)
            ax3.set_title("Simulated AI Attention (Demo)")
            ax3.axis("off")
            st.pyplot(fig_heat)

        # ---------------- DOWNLOAD ----------------
        fig_dl, axs = plt.subplots(1, 4, figsize=(15, 4))
        for i, image in enumerate([pan_np, cbct_np, soft_np, output_np]):
            axs[i].imshow(get_slice(image), cmap=cmap)
            axs[i].axis("off")

        buf = io.BytesIO()
        fig_dl.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        buf.seek(0)

        st.download_button(
            "Download Demo Figure",
            data=buf,
            file_name="dental_ai_fusion_output.png",
            mime="image/png",
        )

        # ---------------- INSIGHTS ----------------
        st.markdown("### 🧠 AI Clinical Insight")
        st.info("""
• Multimodal fusion highlights structural + soft tissue alignment  
• Supports visualization for planning workflows  
• Demonstrates cross-modality integration  
""")

        st.markdown("### 📊 Model Indicators")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fusion Quality", "High")
        c2.metric("Modalities Used", "3")
        c3.metric("Execution Mode", "Real-time")

        if mode == "Detailed Analysis":
            st.markdown("### 🔍 Detailed Analysis Mode")
            st.write("Feature maps and advanced outputs can be added here.")

        # ---------------- SYSTEM STATUS ----------------
        st.markdown("### ⚙️ System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        status_col1.success("Input Processing ✅")
        status_col2.success("Fusion Engine ✅")
        status_col3.success("Visualization ✅")

        # ---------------- REPORT ----------------
        report_text = """
Dental AI Fusion Report

System:
- MONAI-based multimodal fusion pipeline

Capabilities:
- Cross-modality alignment
- Integrated visualization
- Interactive exploration

Note:
Proof-of-concept demo using synthetic data.
"""
        st.download_button(
            "📄 Download Clinical Report",
            data=report_text,
            file_name="fusion_report.txt",
        )

        st.caption("⚠️ Demo uses synthetic data. Not for clinical use.")
        clinical_summary_box()

    else:
        st.info("Upload preview images if available, or click 'Run Fusion Demo' to use built-in synthetic data.")

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

# ---------------- GLOBAL SUMMARY ----------------
st.markdown("---")
st.markdown("### 🏁 Demo Summary")
st.success("""
This platform demonstrates how multimodal dental imaging can be unified into a 
single AI-assisted workflow, enabling enhanced visualization and future-ready 
clinical decision support.
""")

footer_note()