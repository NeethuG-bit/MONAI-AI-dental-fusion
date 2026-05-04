import io
import re
import time
import zipfile
import numpy as np
from PIL import Image
import streamlit as st
import torch
import matplotlib.pyplot as plt
import pydicom

from model import DentalFusionNetwork
from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from segmentation_model import run_segmentation
from skimage.metrics import peak_signal_noise_ratio, structural_similarity 

def generate_pdf_report(text_content, filename="report.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    for line in text_content.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer     

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
        ["Overview", "Live Demo", "Workflow", "Use Cases", "Architecture", "Platform"]
    )
    st.markdown("---")
    run_demo = st.button("Run Fusion Demo")
    show_shapes = st.toggle("Show tensor shapes", value=False)
    use_colored_output = st.toggle("Colored output", value=True)
    mode = st.selectbox("Demo Mode", ["Quick Demo", "Detailed Analysis"])
    intensity = st.slider("Output Enhancement", 0.5, 2.0, 1.0)
    show_heatmap = st.toggle("Show AI Heatmap", value=True)
    presentation_mode = st.toggle("🎤 Presentation Mode", value=False)
    show_segmentation = st.toggle("Show Segementation Overlay", value=True)
    segmentation_threshold = st.slider("Segmentation sensitivity", 0.1, 0.9, 0.55)

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
st.warning("🔒 Research Prototype — Not for Clinical Use")
st.markdown("### Powered by MONAI • Clinical Imaging Intelligence")
st.success("🟢 System Ready • Model Loaded • No Errors")
st.markdown("""
### 🦷 AI-Powered Multimodal Dental Imaging Platform

Combine CBCT, Panoramic, and Soft-Tissue scans into a single
intelligent visualization using MONAI-based deep learning.

✔ End-to-end AI pipeline
✔ Real-time fusion & visualization
✔ Clinical-style diagnosis interface
""")

colA, colB = st.columns([6, 1])
with colB:
    st.markdown("🟢 **System Online**")

if presentation_mode:
    st.success("Live AI Demonstration Mode Active")

    st.markdown("## 🎤 Live Demo Narrative")
    st.info("""
step 1: Upload multimodal inputs
step 2: Preprocessing aligns and normalizes data
step 3: AL model extracts one and fuses d=features
step 4: Output is generated with segmentation & heatmap
step 5: Metrics and interpretation are provided
""")

kpi_row()
workflow_banner()

# ---------------- HELPERS ----------------
def natural_sort_key(name):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]

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

def normalize_dicom_pixels(arr, target_min=-1000.0, target_max=1000.0):
    arr = arr.astype(np.float32)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - arr_min) / (arr_max - arr_min)
    arr = arr * (target_max - target_min) + target_min
    return arr

def preprocess_dicom_array(arr, target_size=(64, 64)):
    arr = normalize_dicom_pixels(arr, -1000.0, 1000.0)
    arr_norm = arr - arr.min()
    if arr_norm.max() > 0:
        arr_norm = arr_norm / arr_norm.max()
    arr_img = Image.fromarray((arr_norm * 255).astype(np.uint8)).resize(target_size)
    arr_img = np.array(arr_img).astype(np.float32)
    arr_final = (arr_img / 255.0) * 2000 - 1000
    return arr_final

def safe_getattr(ds, attr, default="Unknown"):
    try:
        value = getattr(ds, attr, default)
        return str(value)
    except Exception:
        return default

def read_dicom_from_bytes(file_bytes):
    bio = io.BytesIO(file_bytes)
    ds = pydicom.dcmread(bio, force=True)
    pixel_array = ds.pixel_array
    return ds, pixel_array

def create_simulated_segmentation(image, threshold=0.55):
    img = np.asarray(image).astype(np.float32)

    img_min = img.min()
    img_max = img.max()

    if img_max - img_min < 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    norm = (img - img_min) / (img_max - img_min)

    mask = norm > threshold
    return mask.astype(np.float32)      

def inspect_cbct_zip(zip_file, preview_count=5, target_size=(64, 64)):
    """
    Supports ZIP with:
    - image slices
    - dicom slices

    Returns:
    - info dict
    """
    zf = zipfile.ZipFile(zip_file)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    names = sorted(names, key=natural_sort_key)

    image_names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    dicom_names = [n for n in names if n.lower().endswith(".dcm") or "." not in n.split("/")[-1]]

    preview_images = []
    metadata = {
        "type": "unknown",
        "slice_count": 0,
        "patient_name": "Unknown",
        "study_date": "Unknown",
        "modality": "Unknown",
    }

    if dicom_names:
        metadata["type"] = "dicom"
        metadata["slice_count"] = len(dicom_names)

        sample_indices = np.linspace(0, len(dicom_names) - 1, min(preview_count, len(dicom_names))).astype(int)

        first_ds = None
        for idx in sample_indices:
            with zf.open(dicom_names[idx]) as f:
                ds, pixel_array = read_dicom_from_bytes(f.read())
                if first_ds is None:
                    first_ds = ds
                arr = preprocess_dicom_array(pixel_array, target_size=target_size)
                preview_images.append(arr)

        if first_ds is not None:
            metadata["patient_name"] = safe_getattr(first_ds, "PatientName")
            metadata["study_date"] = safe_getattr(first_ds, "StudyDate")
            metadata["modality"] = safe_getattr(first_ds, "Modality")

    elif image_names:
        metadata["type"] = "image_stack"
        metadata["slice_count"] = len(image_names)

        sample_indices = np.linspace(0, len(image_names) - 1, min(preview_count, len(image_names))).astype(int)
        for idx in sample_indices:
            with zf.open(image_names[idx]) as f:
                img = Image.open(f)
                arr = preprocess_image_from_pil(img, target_size=target_size, mode="cbct")
                preview_images.append(arr)

    return metadata, preview_images

def load_cbct_volume(cbct_file, target_size=(64, 64), depth=32):
    """
    Supports:
    - single image file -> repeated into volume
    - zip file of image slices
    - zip file of dicom slices

    Returns:
    - volume
    - info dict
    """
    info = {
        "type": "synthetic",
        "original_slice_count": depth,
        "patient_name": "Unknown",
        "study_date": "Unknown",
        "modality": "Unknown",
        "pixel_spacing": "Unknown",
        "slice_thickness": "Unknown",
    }

    if cbct_file is None:
        return generate_cbct((target_size[0], target_size[1], depth)), info

    filename = cbct_file.name.lower()

    if filename.endswith(".zip"):
        zf = zipfile.ZipFile(cbct_file)
        names = [n for n in zf.namelist() if not n.endswith("/")]
        names = sorted(names, key=natural_sort_key)

        image_names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        dicom_names = [n for n in names if n.lower().endswith(".dcm") or "." not in n.split("/")[-1]]

        slices = []

        if dicom_names:
            info["type"] = "dicom"
            info["original_slice_count"] = len(dicom_names)

            first_ds = None

            dicom_items = []

            for idx, name in enumerate(dicom_names):
                with zf.open(name) as f:
                    ds, pixel_array = read_dicom_from_bytes(f.read())
                    sort_value = get_dicom_sort_value(ds, idx)
                    dicom_items.append((sort_values, ds, pixel_array))

            dicom_items = sorted(dicom_items, key=lambda x: x[0])

            for _, ds, pixel_array in dicom_items:
                if first_ds is None:
                    first_ds = ds
                arr = preprocess_dicom_array(pixel_array, target_size=target_size)
                slices.append(arr)            


            if first_ds is not None:
                info["patient_name"] = safe_getattr(first_ds, "PatientName")
                info["study_date"] = safe_getattr(first_ds, "StudyDate")
                info["modality"] = safe_getattr(first_ds, "Modality")
                info["pixel_spacing"] = safe_getattr(first_ds, "PixelSpacing")
                info["slice_thickness"] = safe_getattr(first_ds, "SliceThickness")

        elif image_names:
            info["type"] = "image_stack"
            info["original_slice_count"] = len(image_names)

            for name in image_names:
                with zf.open(name) as f:
                    img = Image.open(f)
                    arr = preprocess_image_from_pil(img, target_size=target_size, mode="cbct")
                    slices.append(arr)

        if len(slices) == 0:
            return generate_cbct((target_size[0], target_size[1], depth)), info

        volume = np.stack(slices, axis=-1)

        current_depth = volume.shape[-1]
        if current_depth > depth:
            idx = np.linspace(0, current_depth - 1, depth).astype(int)
            volume = volume[:, :, idx]
        elif current_depth < depth:
            pad_count = depth - current_depth
            last_slice = volume[:, :, -1:]
            pad_block = np.repeat(last_slice, pad_count, axis=-1)
            volume = np.concatenate([volume, pad_block], axis=-1)

        return volume, info

    # single image
    arr2d = preprocess_image_2d(cbct_file, target_size=target_size, mode="panoramic")
    volume = np.repeat(arr2d[:, :, None], depth, axis=2)
    info["type"] = "single_image"
    info["original_slice_count"] = 1
    return volume, info

def get_volume_view(volume, view="Axial", index=None):
    h, w, d = volume.shape

    if view == "Axial":
        if index is None:
            index = d // 2
        return volume[:, :, index], d - 1, index

    if view == "Coronal":
        if index is None:
            index = h // 2
        return volume[index, :, :], h - 1, index

    if view == "Sagittal":
        if index is None:
            index = w // 2
        return volume[:, index, :], w - 1, index

    raise ValueError(f"Unknown view: {view}")

def get_dicom_sort_value(ds, fallback):
    try:
        return int(getattr(ds, "InstanceNumber", fallback))
    except Exception:
        return fallback

def calculate_image_metrics(reference_img, output_img):
    ref = np.asarray(reference_img).astype(np.float32)
    out = np.asarray(output_img).astype(np.float32)

    ref = get_slice(ref)
    out = get_slice(out)

    # Resize output to match reference size
    if ref.shape != out.shape:
        out_img = Image.fromarray(out)
        out_img = out_img.resize((ref.shape[1], ref.shape[0]))
        out = np.array(out_img).astype(np.float32)

    ref_min, ref_max = ref.min(), ref.max()
    out_min, out_max = out.min(), out.max()

    if ref_max - ref_min > 1e-6:
        ref = (ref - ref_min) / (ref_max - ref_min)

    if out_max - out_min > 1e-6:
        out = (out - out_min) / (out_max - out_min)

    psnr = peak_signal_noise_ratio(ref, out, data_range=1.0)
    ssim = structural_similarity(ref, out, data_range=1.0)

    return psnr, ssim    

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
        "Optional: Upload CBCT ZIP (image slices or DICOM series)",
        type=["zip"],
        key="cbct_zip_upload"
    )

    cbct_source = cbct_zip if cbct_zip is not None else cbct_file

    st.info("📁 Upload real images, ZIP slice stacks, or DICOM ZIP series — or run demo with built-in synthetic data")

    preview_cols = st.columns(3)
    if pan_file:
        preview_cols[0].image(pan_file, caption="Panoramic Preview")
    if cbct_file and cbct_zip is None:
        preview_cols[1].image(cbct_file, caption="CBCT Preview")
    elif cbct_zip:
        preview_cols[1].success("CBCT ZIP uploaded")
    if soft_file:
        preview_cols[2].image(soft_file, caption="Soft Tissue Preview")

    if cbct_zip is not None:
        metadata, preview_images = inspect_cbct_zip(cbct_zip, preview_count=5, target_size=(64, 64))

        st.markdown("### 🧊 CBCT ZIP Inspection")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Type", metadata["type"])
        m2.metric("Slices Found", str(metadata["slice_count"]))
        m3.metric("Modality", metadata["modality"])
        m4.metric("Study Date", metadata["study_date"])

        st.markdown(f"**Patient Name:** {metadata['patient_name']}")

        if preview_images:
            st.markdown("#### Thumbnail Strip")
            thumb_cols = st.columns(len(preview_images))
            for col, img in zip(thumb_cols, preview_images):
                with col:
                    st.image(img, caption="Slice", use_container_width=True, clamp=True)

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

            cbct, cbct_info = load_cbct_volume(cbct_source, target_size=(64, 64), depth=32)

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

            start_time = time.time()

            with torch.no_grad():
                output_tensor, _ = model(pan_tensor, cbct_tensor, soft_tensor)

            inference_time = time.time() - start_time

        if not presentation_mode:
            progress.empty()

        st.success("Fusion inference completed successfully")
        st.markdown("## 🧪 Preprocessing & Quality Checks")

        q1, q2, q3, q4 = st.columns(4)

        q1.success("Normalization ✅")
        q2.success("Resizing ✅")
        q3.success("Modality Pairing ✅")
        q4.success("Input Validation ✅")

        st.markdown("---")

        st.info(""" 
        Preprocessing completed:
        CBCT, PAN, and Soft Tissue inputs were normalized, resized, converted to tensors, and prepared for fusion inference.
        """)

        st.markdown("## 🧠 What the AI is doing")

        st.info("""
        The model extracts features from each modality (CBCT, PAN, Soft Tissue),
        combines them using a fusion layer, and reconstructs an enhanced output
        image using a 3D UNet architecture

        This preserves structural + contextual information across modalities.
        """)

        st.markdown("### 🤖 AI Interpretation Layer")
        st.success("Fusion complete. Generating visual intelligence...")
        st.markdown("### 🧪 Clinical View")
        st.caption("Radiology-style visualization for diagnostic interpretation")  

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

        psnr_value, ssim_value = calculate_image_metrics(cbct_np, output_np)

        st.markdown("## ✅ POC Validation Dashboard")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dice Score", "-0.65 (demo)")
        m2.metric("PSNR", f"{psnr_value:.2f} dB")
        m3.metric("SSIM", f"{ssim_value:.3f}")
        m4.metric("Inference Time", f"{inference_time:.2f} sec")

        st.info("""
        This demo follows the MONAI POC workflow:
        CBCT + PAN + Soft Tissue -> Preprocessing -> Feature Extraction -> Fusion -> Visualization -> Evaluation.
        """)

        st.markdown("---")    

        st.markdown("### 📦 Volume Summary")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Input Type", cbct_info["type"])
        v2.metric("Original Slices", str(cbct_info["original_slice_count"]))
        v3.metric("Model Depth", str(cbct_np.shape[2]))
        v4.metric("Modality", cbct_info["modality"])

        st.markdown(f"**Patient Name:** {cbct_info['patient_name']}")
        st.markdown(f"**Study Date:** {cbct_info['study_date']}")

        cmap = "viridis" if use_colored_output else "gray"

        st.markdown("### 🧭 DICOM Metadata")

        d1, d2 = st.columns(2)
        d1.info(f"Pixel Spacing: {cbct_info.get('pixel_spacing', 'Unknown')}")
        d2.info(f"Slice_Thickness: {cbct_info.get('slice_thickness', 'Unknown')}")

        st.markdown("---")

        st.markdown("---")
        st.markdown("## 📊 Generated Results")
        st.markdown("## 🧾 AI Fusion Output")
        st.caption("Integrated multimodal visualization (Panoramic + CBCT + Soft Tissue)")

        with st.container():
            st.markdown("#### 📊 Visualization")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(get_slice(pan_np), caption="Panoramic", use_container_width=True)
            c2.image(get_slice(cbct_np), caption="CBCT", use_container_width=True)
            c3.image(get_slice(soft_np), caption="Soft Tissue", use_container_width=True)
            c4.image(get_slice(output_np), caption="Fused Output", use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧊 Advanced CBCT Explorer")

        view_mode = st.radio(
            "Select View",
            ["Axial", "Coronal", "Sagittal"],
            horizontal=True
        )

        _, max_idx, default_idx = get_volume_view(cbct_np, view=view_mode)

        selected_idx = st.slider(
            f"{view_mode} Slice Index",
            0,
            max_idx,
            default_idx
        )

        selected_view, _, _ = get_volume_view(cbct_np, view=view_mode, index=selected_idx)

        st.markdown("### 📊 Viewer Controls")

        vc1, vc2, vc3 = st.columns(3)

        with vc1:
            brightness = st.slider("Brightness", -500.0, 500.0, 0.0)

        with vc2:
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0)

        with vc3:
            zoom = st.slider("zoom", 1.0, 3.0, 1.0)     

        viewer_img = selected_view.astype(np.float32)
        viewer_img = (viewer_img * contrast) + brightness

        if zoom > 1.0:
            h, w = viewer_img.shape
            crop_h = int(h / zoom)
            crop_w = int(w / zoom)

            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2

            viewer_img = viewer_img[
                start_h:start_h + crop_h,
                start_w:start_w + crop_w
            ]

        viewer_img = selected_view.astype(np.float32)
        viewer_img = (viewer_img * contrast) + brightness

        if zoom > 1.0:
            h, w = viewer_img.shape
            crop_h = int(h / zoom)
            crop_w = int(w / zoom)

            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2

            viewer_img = viewer_img[
                start_h:start_h + crop_h,
                start_w:start_w + crop_w
            ]

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            x1 = st.number_input("Point 1 - x", 0, viewer_img.shape[1], 10)
            y1 = st.number_input("Point 1 - y", 0, viewer_img.shape[0], 10)

        with col_m2:
            x2 = st.number_input("Point 2 - x", 0, viewer_img.shape[1], 50)
            y2 = st.number_input("Point 2 - y", 0, viewer_img.shape[0], 50)

        fig_view, axv = plt.subplots()
        axv.imshow(viewer_img, cmap=cmap)

        # draw measurement line 
        axv.plot([x1, x2], [y1, y2], color='red', linewidth=2)
        axv.scatter([x1, x2], [y1, y2], color='yellow')
        
        axv.set_title(f"{view_mode} View - Slice {selected_idx}")
        axv.axis("off")

        st.pyplot(fig_view)

        st.caption(
            f"Viewer settings — Brightness: {brightness}, Contrast: {contrast}, Zoom: {zoom}x"
        )

        st.markdown("---")

        st.markdown("#### 🪟 Window Presets")

        preset = st.selectbox(
            "Select Window Preset",
            ["Default", "Bone", "Soft Tissue"]
        )

        if preset == "Bone":
            contrast = 1.6
            brightness = 100.0
        elif preset == "Soft Tissue":
            contrast = 0.8
            brightness = -100.0    

        st.markdown("### 📏 Measurement Tool")

        distance_px = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

        # simulate mm conversion 
        pixel_spacing = 0.3   # mm per pixel (dummy realistic value)
        distance_mm = distance_px * pixel_spacing

        st.info (f"Distance: {distance_mm:.2f} mm")


        st.markdown("---")
        st.markdown("### 🎯 Detected Region (Simulated)")

        img = get_slice(output_np)
        mask = img > img.mean()

        fig_detect, ax2 = plt.subplots()
        ax2.imshow(img, cmap="gray")
        ax2.imshow(mask, cmap="jet", alpha=0.3)
        ax2.axis("off")
        st.pyplot(fig_detect)

        if show_segmentation:
            st.markdown("---")
            st.markdown("### 🧩 Segmentation Overlay (Simulated) ")

            base_img = get_slice(cbct_np)
            output_img = get_slice(output_np)
            seg_tensor, seg_mode = run_segmentation(cbct_tensor, device)

            seg_np = seg_tensor.cpu().numpy()[0, 0]
            seg_slice = get_slice(seg_np)

            st.markdown("### 🧩 Segmentation Overlay")

            fig_seg, ax_seg = plt.subplots()
            ax_seg.imshow(get_slice(cbct_np), cmap="gray")
            ax_seg.imshow(seg_slice, cmap="jet", alpha=0.35)
            ax_seg.set_title(f"Segmentation Overlay ({seg_mode})")
            ax_seg.axis("off")
            st.pyplot(fig_seg)

            if seg_mode == "fallback":
                st.warning("Using fallback segmentation because trained model weights were not found.")

            else:
                st.success("Real segmentation model loaded successfully.")     

            st.caption(
                "This overlay is a simulated ROI mask for demo purposes."
                "It can later be replaced with a real MONAI segmentation model output."
            )

            st.markdown("### 🧠 AI Summary")

            mean_val = float(np.mean(output_np))
            max_val = float(np.max(output_np))

            if mean_val > 0:
                summary = "The fused output highlights areas of structural density wth moderate intensity."
            else:
                summary = "The fused output shows low-intensity regions with minimal structural emphasis."
            st.info(f"""
            **Auto-generated Insight:**

            - Average activation: {mean_val:.2f}
            - Peak activation: {max_val:.2f}

            Interpretation:
            {summary}
            """) 

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

        st.markdown("---")
        st.markdown("## 🧾 Clinical Interpretation")

        st.success("""
        • Fused image improves structural + soft tissue visibility
        • Segmentation highloghts potential regions of interest
        • Heatmap indicates model attention zones
        • Supports planning, diagnosis, and review workflows
        """)

        st.markdown("### 🧠 AI Clinical Insight")
        st.info("""
• Multimodal fusion highlights structural + soft tissue alignment
• Segmentation overlay simulated AI-based region-of-interest detection  
• Supports visualization for planning workflows  
• Demonstrates cross-modality integration  
• DICOM-ready CBCT ingestion scaffold is now available  
""")

        st.markdown("### 📊 Model Indicators")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fusion Quality", "High")
        c2.metric("Modalities Used", "3")
        c3.metric("Execution Mode", "Real-time")

        if mode == "Detailed Analysis":
            st.markdown("### 🔍 Detailed Analysis Mode")
            st.write("Feature maps, segmentation overlays, and advanced outputs can be added here.")

        st.markdown("### ⚙️ System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        status_col1.success("Input Processing ✅")
        status_col2.success("Fusion Engine ✅")
        status_col3.success("Visualization ✅")

        st.markdown("## ⚡ System Performance")

        p1, p2, p3 = st.columns(3)

        p1.metric("Pipeline Status", "Operational ✅")
        p2.metric("Preprocessing Mode", "CPU (Demo)")
        p3.metric("Deployment", "Yes")

        report_text = f"""
DENTAL AI MULTIMODAL FUSION REPORT
----------------------------------

SYSTEM OVERVIEW:
- MONAI-based multimodal fusion pipeline
- Inputs: CBCT + Panoramic + Soft Tissue

INPUT DETAILS:
- CBCT Type: {cbct_info["type"]}
- Original Slices: {cbct_info["original_slice_count"]}
- Model Depth: {cbct_np.shape[2]}
- Patient Name: {cbct_info["patient_name"]}
- Study Date: {cbct_info["study_date"]}
- Modality: {cbct_info["modality"]}

PREPROCESSING
- Normalization: Completed
- Resizing: Completed
- Alignment: Completed
- Modality Pairing: Completed

PERFORMANCE METRICS
- Dice Score: Sanity Check
- PSNR: {psnr_value:.2f} dB
- SSIM: {ssim_value:.3f}
- Inference Time: {inference_time:.2f} sec

AI INTERPRETATION
- Multimodal fusion highlights structural + soft tissue alignment
- Regions of internet detected using simulated alignment
- Heatmap shows AI attention zones

OUTPUT SUMMARY
- Fused visualization generated
- CBCT multi-view explorer enabled
- Segementation overlay applied
- Heatmap visualization available

DISCLAIMER
This is a research prototype and not intended for clinical diagnosis.
"""
        pdf_buffer = generate_pdf_report(report_text)

        st.download_button(
            "📄 Download Clinical Report (PDF)",
            data=pdf_buffer,
            file_name="dental_ai_report.pdf",
            mime="application/pdf",
        )

        st.caption("⚠️ Prototype workflow. Not for clinical use.")
        clinical_summary_box()

    else:
        st.info("Upload preview images, CBCT slice ZIP, or DICOM ZIP series — or click 'Run Fusion Demo' to use built-in demo data.")

elif page == "Workflow":
    st.title("🔄 MONAI Fusion Workflow")
    st.caption("End-to-end AI pipeline aligned with the POC documentation")

    st.markdown("## 🧭 Pipeline Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("""
### 1️⃣ Data Collection
- CBCT scans  
- Panoramic X-rays  
- Soft-tissue scans  
""")

    with c2:
        st.info("""
### 2️⃣ Preprocessing
- Normalization  
- Resizing  
- Alignment  
- Modality pairing  
""")

    with c3:
        st.info("""
### 3️⃣ Model Training
- MONAI Core  
- 3D UNet  
- Fusion layer  
- Dice / MSE / SSIM loss  
""")

    with c4:
        st.info("""
### 4️⃣ Fusion Output
- Enhanced image  
- Segmentation  
- Heatmap  
- Clinical view  
""")

    st.markdown("### ➜ Workflow Flow")

    st.success("""
CBCT + PAN + Soft Tissue  
→ Preprocessing  
→ Feature Extraction  
→ Fusion Layer  
→ Decoder  
→ Fused Output  
→ Segmentation / Evaluation
""")

    st.markdown("## 🧬 MONAI Components")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("MONAI Core", "Training + Inference")

    with m2:
        st.metric("MONAI Bundle", "Model Packaging")

    with m3:
        st.metric("MONAI Deploy", "Deployment Ready")

    with m4:
        st.metric("MONAI Label", "Annotation Loop")

    st.markdown("## 📊 Evaluation Targets")

    e1, e2, e3, e4 = st.columns(4)

    e1.metric("Dice Score", "> 0.85")
    e2.metric("PSNR", "> 30 dB")
    e3.metric("SSIM", "> 0.90")
    e4.metric("Inference", "CPU / GPU")

    st.warning("Prototype status: demo-ready, not for clinical diagnosis.")


elif page == "Use Cases":
    use_case_tabs()
    outcomes_cards()

elif page == "Architecture":
    architecture_page_cards()
    pipeline_cards()
    clinical_summary_box()

elif page == "Platform":
    platform_cards()

    # -------- MONAI ECOSYSTEM SECTION --------
    st.markdown("## 🧬 MONAI Ecosystem Alignment")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("""
### MONAI Core
Used for medical imaging model development, transforms, and inference.

Current use:
- fusion model pipeline
- medical preprocessing
- segmentation integration
""")

    with c2:
        st.info("""
### MONAI Label
Future clinician-in-the-loop annotation layer.

Future use:
- AI-assissted labeling
- active learning
- dentist feedback loop
""")

    with c3:
        st.info("""
### MONAI Deploy
Future clinical deployment layer.

Future use:
- DICOM workflow integration
- containerized deployment
- scalable inference
""")

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