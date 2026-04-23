import io
import time
import zipfile
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

if "panel" not in st.session_state:
    st.session_state["panel"] = "pan"

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
def get_slice(img):
    squeezed = np.asarray(img).squeeze()
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3:
        mid = squeezed.shape[-1] // 2
        return squeezed[:, :, mid]
    raise ValueError(f"Unsupported array shape: {squeezed.shape}")

def preprocess_image_2d(uploaded_file, target_size=(64, 64), mode="panoramic"):
    img = Image.open(uploaded_file).convert("L").resize(target_size)
    arr = np.array(img).astype(np.float32)

    if mode == "panoramic":
        arr = (arr / 255.0) * 2000 - 1000
    elif mode == "soft":
        arr = (arr / 255.0) * 15.0
    else:
        arr = arr / 255.0

    return arr

def preprocess_image_from_pil(img, target_size=(64, 64), mode="cbct"):
    img = img.convert("L").resize(target_size)
    arr = np.array(img).astype(np.float32)

    if mode == "cbct":
        arr = (arr / 255.0) * 2000 - 1000
    else:
        arr = arr / 255.0

    return arr

def inspect_cbct_zip(zip_file, preview_count=3, target_size=(64, 64)):
    """
    Returns:
    - slice_count
    - preview_images (list of 2D arrays)
    """
    zf = zipfile.ZipFile(zip_file)
    names = sorted([
        n for n in zf.namelist()
        if n.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    preview_images = []
    if names:
        sample_indices = np.linspace(0, len(names) - 1, min(preview_count, len(names))).astype(int)
        for idx in sample_indices:
            with zf.open(names[idx]) as f:
                img = Image.open(f)
                arr = preprocess_image_from_pil(img, target_size=target_size, mode="cbct")
                preview_images.append(arr)

    return len(names), preview_images

def load_cbct_volume(cbct_file, target_size=(64, 64), depth=32):
    """
    Supports:
    - single image file -> repeated into volume
    - zip file of slice images -> stacked into volume

    Returns:
    - volume
    - original_slice_count
    """
    if cbct_file is None:
        return generate_cbct((target_size[0], target_size[1], depth)), depth

    filename = cbct_file.name.lower()

    if filename.endswith(".zip"):
        zf = zipfile.ZipFile(cbct_file)
        names = sorted([
            n for n in zf.namelist()
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        slices = []
        for name in names:
            with zf.open(name) as f:
                img = Image.open(f)
                arr = preprocess_image_from_pil(img, target_size=target_size, mode="cbct")
                slices.append(arr)

        if len(slices) == 0:
            return generate_cbct((target_size[0], target_size[1], depth)), depth

        volume = np.stack(slices, axis=-1)
        original_slice_count = volume.shape[-1]

        current_depth = volume.shape[-1]
        if current_depth > depth:
            idx = np.linspace(0, current_depth - 1, depth).astype(int)
            volume = volume[:, :, idx]
        elif current_depth < depth:
            pad_count = depth - current_depth
            last_slice = volume[:, :, -1:]
            pad_block = np.repeat(last_slice, pad_count, axis=-1)
            volume = np.concatenate([volume, pad_block], axis=-1)

        return volume, original_slice_count

    arr2d = preprocess_image_2d(cbct_file, target_size=target_size, mode="panoramic")
    volume = np.repeat(arr2d[:, :, None], depth, axis=2)
    return volume, 1

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
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📖 Imaging Modalities")
    st.markdown(
        """
🦷 **Panoramic (OPG)** <span title='Full jaw 2D X-ray showing teeth and jaw structure'>ℹ️</span>  

🧊 **CBCT** <span title='3D cone beam CT scan for volumetric dental imaging'>ℹ️</span>  

🧠 **Soft Tissue** <span title='Facial surface / soft tissue structure for esthetic planning'>ℹ️</span>  

🔗 **Fusion** <span title='Combines multiple imaging modalities into one AI output'>ℹ️</span>  
""",
        unsafe_allow_html=True
    )

    st.markdown("### 📖 Click a term to learn more")
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        if st.button("🦷 Panoramic"):
            st.session_state["panel"] = "pan"
    with g2:
        if st.button("🧊 CBCT"):
            st.session_state["panel"] = "cbct"
    with g3:
        if st.button("🧠 Soft Tissue"):
            st.session_state["panel"] = "soft"
    with g4:
        if st.button("🔗 Fusion"):
            st.session_state["panel"] = "fusion"

    pan_file, cbct_file, soft_file = upload_console()

    cbct_zip = st.file_uploader(
        "Optional: Upload CBCT slice ZIP",
        type=["zip"],
        key="cbct_zip_upload"
    )

    cbct_source = cbct_zip if cbct_zip is not None else cbct_file

    st.info("📁 Upload real images or run demo with built-in synthetic data")

    preview_cols = st.columns(3)
    if pan_file:
        preview_cols[0].image(pan_file, caption="Panoramic Preview")
    if cbct_file and cbct_zip is None:
        preview_cols[1].image(cbct_file, caption="CBCT Preview")
    elif cbct_zip:
        preview_cols[1].success("CBCT ZIP uploaded")
    if soft_file:
        preview_cols[2].image(soft_file, caption="Soft Tissue Preview")

    # -------- STAGE 2 ZIP INSPECTION --------
    if cbct_zip is not None:
        slice_count, preview_images = inspect_cbct_zip(cbct_zip, preview_count=3, target_size=(64, 64))

        st.markdown("### 🧊 CBCT ZIP Inspection")
        s1, s2 = st.columns([1, 2])
        with s1:
            st.metric("Slices Found", str(slice_count))
        with s2:
            st.success("ZIP recognized as CBCT slice stack")

        if preview_images:
            st.markdown("#### Preview Slices")
            preview_slice_cols = st.columns(len(preview_images))
            for col, img in zip(preview_slice_cols, preview_images):
                with col:
                    st.image(img, caption="Slice Preview", use_container_width=True, clamp=True)

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

            if pan_file:
                pan = preprocess_image_2d(pan_file, target_size=(64, 64), mode="panoramic")
            else:
                pan = generate_panoramic((64, 64))

            cbct, original_cbct_depth = load_cbct_volume(cbct_source, target_size=(64, 64), depth=32)

            if soft_file:
                soft = preprocess_image_2d(soft_file, target_size=(64, 64), mode="soft")
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
        st.markdown("### 🤖 AI Interpretation Layer")
        st.success("Fusion complete. Generating visual intelligence...")

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

        st.markdown("### 📦 Volume Summary")
        v1, v2, v3 = st.columns(3)
        v1.metric("Original CBCT Slices", str(original_cbct_depth))
        v2.metric("Model Volume Depth", str(cbct_np.shape[2]))
        v3.metric("Input Mode", "Real Upload" if cbct_source is not None else "Synthetic Demo")

        cmap = "viridis" if use_colored_output else "gray"

        st.markdown("---")
        st.markdown("## 📊 Generated Results")
        st.markdown("## 🧾 AI Fusion Results")
        st.caption("Multimodal visualization output from the fusion engine")

        with st.container():
            st.markdown("#### 📊 Visualization")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(get_slice(pan_np), caption="Panoramic", use_container_width=True)
            c2.image(get_slice(cbct_np), caption="CBCT", use_container_width=True)
            c3.image(get_slice(soft_np), caption="Soft Tissue", use_container_width=True)
            c4.image(get_slice(output_np), caption="Fused Output", use_container_width=True)

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

        st.markdown("---")
        st.markdown("### 🎯 Detected Region (Simulated)")

        img = get_slice(output_np)
        mask = img > img.mean()

        fig_detect, ax2 = plt.subplots()
        ax2.imshow(img, cmap="gray")
        ax2.imshow(mask, cmap="jet", alpha=0.3)
        ax2.axis("off")
        st.pyplot(fig_detect)

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

        st.markdown("### ⚙️ System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        status_col1.success("Input Processing ✅")
        status_col2.success("Fusion Engine ✅")
        status_col3.success("Visualization ✅")

        report_text = f"""
Dental AI Fusion Report

System:
- MONAI-based multimodal fusion pipeline

Capabilities:
- Cross-modality alignment
- Integrated visualization
- Interactive exploration

CBCT Details:
- Original uploaded slices: {original_cbct_depth}
- Model depth after preprocessing: {cbct_np.shape[2]}

Note:
Proof-of-concept demo using real uploaded images when provided, otherwise synthetic data.
"""
        st.download_button(
            "📄 Download Clinical Report",
            data=report_text,
            file_name="fusion_report.txt",
        )

        st.caption("⚠️ Prototype workflow. Not for clinical use.")
        clinical_summary_box()

    else:
        st.info("Upload preview images if available, or click 'Run Fusion Demo' to use built-in demo data.")

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

st.markdown("---")
st.markdown("### 🏁 Demo Summary")
st.success("""
This platform demonstrates how multimodal dental imaging can be unified into a
single AI-assisted workflow, enabling enhanced visualization and future-ready
clinical decision support.
""")

footer_note()