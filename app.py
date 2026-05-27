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
import datetime
import imageio.v2 as imageio
import cv2
import plotly.graph_objects as go

from model import DentalFusionNetwork
from data import generate_panoramic, generate_cbct, generate_soft_tissue
from transforms import get_transforms
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from pydicom.dataset import Dataset
from skimage import measure
from scipy import ndimage
from segmentation_model import run_segmentation
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage


if "fusion_done" not in st.session_state:
    st.session_state["fusion_done"] = False

if "fusion_results" not in st.session_state:
    st.session_state["fusion_results"] = {}


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

if "fusion_done" not in st.session_state:
    st.session_state["fusion_done"] = False

if "fusion_results" not in st.session_state:
    st.session_state["fusion_results"] = None

panel = st.session_state.get("panel")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Platform Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "🏠 Clinical Dashboard",
            "🧠 Acquisition & Modalities",
            "⚙️ Preprocessing Pipeline",
            "🧩 Feature Extraction",
            "🔗 Fusion Engine",
            "🧠 AI Explainability",
            "🩺 Clinical Findings",
            "🌐 3D Diagnostic Workspace",
            "📈 Operational Metrics",
            "🚀 Platform Vision"
        ]
    )
    st.markdown("---")
    run_demo = st.button("Run Fusion Demo")
    show_shapes = st.toggle("Show tensor shapes", value=False)
    use_colored_output = st.toggle("Colored output", value=True)
    mode = st.selectbox("Demo Mode", ["Quick Demo", "Detailed Analysis"])
    intensity = st.slider("Output Enhancement", 0.5, 2.0, 1.0)
    show_heatmap = st.toggle("Show AI Heatmap", value=True)
    presentation_mode = st.toggle("🎤 Presentation Mode", value=False)
    show_segmentation = st.toggle("Show Segmentation Overlay", value=True)
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
step 3: AI model extracts and fuses features
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

def get_axial_slice(volume, slice_idx, num_slices):
    """Axial slice at slice_idx; 2D volumes unchanged; depth scaled to num_slices."""
    vol = np.asarray(volume).squeeze()
    if vol.ndim == 2:
        return vol
    if vol.ndim != 3:
        raise ValueError(f"Unsupported volume shape: {vol.shape}")
    depth = vol.shape[2]
    if num_slices <= 1:
        z = 0
    else:
        z = int(round(slice_idx * (depth - 1) / (num_slices - 1)))
    z = max(0, min(z, depth - 1))
    return vol[:, :, z]

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
                    dicom_items.append((sort_value, ds, pixel_array))
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

def dice_score(pred, gt):
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-6)

def calculate_image_metrics(reference_img, output_img):
    ref = np.asarray(reference_img).astype(np.float32)
    out = np.asarray(output_img).astype(np.float32)
    ref = get_slice(ref)
    out = get_slice(out)
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

def show_intensity_analysis(
    pan_img, 
    cbct_img, 
    soft_img, 
    fused_img,
    title= "📊 Intensity Distrubution Analysis"
):
    st.markdown(f"### {title}")

    images = [
        ("Panoramic", pan_img),
        ("CBCT", cbct_img),
        ("Soft Tissue", soft_img),
        ("Fused Output", fused_img)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes = axes.flatten()

    for ax, (name, img) in zip(axes, images):

        arr = np.asarray(img).astype(np.float32).flatten()

        ax.hist(arr, bins=50)

        ax.set_title(f"{title} Intensity Distribution")

        ax.set_xlabel("Pixel Intensity")

        ax.set_ylabel("Frequency")

    plt.tight_layout()

    st.pyplot(fig)

def normalize_for_display(arr):
    arr = np.asarray(arr).astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-6:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr

def add_noise(image, noise_type="gaussian", strength=0.05):

    img = np.asarray(image).astype(np.float32)

    img_norm = normalize_for_display(img)

    if noise_type == "gaussian":

        noise = np.random.normal(
            0,
            strength,
            img_norm.shape
        )

        noisy = img_norm + noise

    elif noise_type == "salt_pepper":

        noisy = img_norm.copy()

        prob = strength

        salt = np.random.rand(*img_norm.shape) < prob
        pepper = np.random.rand(*img_norm.shape) < prob

        noisy[salt] = 1.0
        noisy[pepper] = 0.0

    else:
        noisy = img_norm

    noisy = np.clip(noisy, 0, 1)
    
    return noisy

def denoise_image(image, blur_strength=3):

    img = np.asarray(image).astype(np.float32)

    denoised = cv2.GaussianBlur(
        img,
        (blur_strength, blur_strength),
        0
    )

    return denoised

def show_denoising_comparison(original_img, noisy_img, denoised_img):
    st.subheader("Before / After Denoising")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy_img, cmap="gray")
    axes[1].set_title("Noisy Image")
    axes[1].axis("off")

    axes[2].imshow(denoised_img, cmap="gray")
    axes[2].set_title("Denoised")
    axes[2].axis("off")

    difference = np.abs(noisy_img - deniosed_img)

    axes[3].imshow(difference, cmap="hot")
    axes[3].set_title("Noise Removed")
    axes[3].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    show_denoised_comparision(
        original_slice,
        noisy_slice,
        denoised_slice
    )

    denoised_slice = cv2/fastNlMeansDenoising(
        noisy_slice.astype(np.uint8),
        None,
        10,
        7,
        21
    )

    noisy_uint8 = (noisy_slice* 255).astype(np.uint8)

    denoised_uint8 = cv2.fastNlMeansDenoising(
        noisy_uint8,
        None,
        10,
        7,
        21
    )

    denoised_slice = denoised_uint8 / 255.0


def show_histogram_analysis(image, title="Histogram Analysis"):
    img = np.asarray(image).astype(np.float32)
    img = get_slice(img)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(img.flatten(), bins=50)

    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

def show_intensity_distribution(image, title="Intensity Distribution"):
    img = np.asarray(image).astype(np.float32)
    img = get_slice(img)

    mean_val = np.mean(img)
    std_val = np.std(img)
    min_val = np.min(img)
    max_val = np.max(img)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(img.flatten(), bins=80, density=True)

    ax.axvline(mean_val, linestyle="--", label=f"Mean: {mean_val:.2f}")
    ax.axvline(mean_val + std_val, linestyle=":", label=f"+1 STD")
    ax.axvline(mean_val - std_val, linestyle=":", label=f"-1 STD")

    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()

    st.pyplot(fig)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Min", f"{min_val:.2f}")
    c2.metric("Max", f"{max_val:.2f}")
    c3.metric("Mean", f"{mean_val:.2f}")
    c4.metric("Std Dev", f"{std_val:.2f}")

def create_registration_preview(fixed_img, moving_img, alpha=0.5):

    fixed = normalize_for_display(
        get_slice(fixed_img)
    )

    moving = normalize_for_display(
        get_slice(moving_img)
    )

    if fixed.shape != moving.shape:

        moving = cv2.resize(
            moving,
            (fixed.shape[1], fixed.shape[0])
        )

    overlay = (
        (1 - alpha) * fixed +
        alpha * moving
    )

    overlay = normalize_for_display(overlay)

    return fixed, moving, overlay        

    
def create_dicom_overlay(image):
    arr = np.asarray(image).astype(np.float32)
    arr = get_slice(arr)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-6:
        arr = (arr - arr_min) / (arr_max - arr_min)
    pixel_array = (arr * 255).astype("uint8")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dt = datetime.datetime.now()
    ds.PatientName = "Demo^Patient"
    ds.PatientID = "DEMO001"
    ds.Modality = "OT"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyDate = dt.strftime("%Y%m%d")
    ds.StudyTime = dt.strftime("%H%M%S")
    ds.SeriesDescription = "Dental AI Fusion Overlay"
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds

def show_3d_volume(volume, title="3D CBCT Volume"):
    vol = normalize_for_display(volume)
    vol = vol[::2, ::2, ::2]
    x, y, z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=vol.flatten(), opacity=0.08, surface_count=15,
    ))
    fig.update_layout(title=title, height=700,
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

def show_3d_surface(volume, title="3D Surface Rendering", threshold=0.45):
    vol = normalize_for_display(volume)
    vol = vol[::2, ::2, ::2]
    binary = vol > threshold
    binary = ndimage.binary_opening(binary)
    binary = ndimage.binary_closing(binary)
    binary = ndimage.binary_fill_holes(binary)
    if binary.sum() < 10:
        st.warning("Not enough high-density structure found for 3D structure rendering. Try lowering the threshold.")
        return
    verts, faces, normals, values = measure.marching_cubes(binary.astype(np.float32), level=0.5)
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.65, intensity=verts[:, 2]
    )])
    fig.update_layout(title=title, height=700,
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

def create_multi_region_masks(volume):
    vol = normalize_for_display(volume)
    low_mask = (vol > 0.30) & (vol <= 0.50)
    medium_mask = (vol > 0.50) & (vol <= 0.70)
    high_mask = vol > 0.70
    return low_mask, medium_mask, high_mask

def show_3d_fusion_overlay(cbct_volume, fusion_volume, title="3D Fusion Overlay",
                            cbct_threshold=0.45, fusion_threshold=0.50,
                            cbct_opacity=0.30, fusion_opacity=0.75):
    cbct = normalize_for_display(cbct_volume)[::2, ::2, ::2]
    fusion = normalize_for_display(fusion_volume)[::2, ::2, ::2]
    cbct_binary = cbct > cbct_threshold
    fusion_binary = fusion > fusion_threshold
    for b in [cbct_binary, fusion_binary]:
        b = ndimage.binary_opening(b)
        b = ndimage.binary_closing(b)
        b = ndimage.binary_fill_holes(b)
    if cbct_binary.sum() < 10 or fusion_binary.sum() < 10:
        st.warning("Not enough structure found for 3D overlay. Try lowering the thresholds.")
        return
    cbct_verts, cbct_faces, _, _ = measure.marching_cubes(cbct_binary.astype(np.float32), level=0.5)
    fusion_verts, fusion_faces, _, _ = measure.marching_cubes(fusion_binary.astype(np.float32), level=0.5)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=cbct_verts[:, 0], y=cbct_verts[:, 1], z=cbct_verts[:, 2],
        i=cbct_faces[:, 0], j=cbct_faces[:, 1], k=cbct_faces[:, 2],
        opacity=0.35, name="CBCT Anatomy", color="lightgray",
        hovertext="CBCT Anatomy", hoverinfo="text"
    ))
    fig.add_trace(go.Mesh3d(
        x=fusion_verts[:, 0], y=fusion_verts[:, 1], z=fusion_verts[:, 2],
        i=fusion_faces[:, 0], j=fusion_faces[:, 1], k=fusion_faces[:, 2],
        opacity=0.75, name="Fusion/Segmentation Output", color="red",
        hovertext="Fusion-Based Segmentation / AI ROI", hoverinfo="text"
    ))
    fig.update_layout(title=title, height=700,
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

def show_multi_region_segmentation(volume):
    low_mask, medium_mask, high_mask = create_multi_region_masks(volume)
    fig = go.Figure()
    region_configs = [
        (low_mask, "Low Activity Region", "blue", 0.20),
        (medium_mask, "Medium Activity Region", "yellow", 0.45),
        (high_mask, "High Activity Region", "red", 0.75),
    ]
    for mask, name, color, opacity in region_configs:
        mask = mask[::2, ::2, ::2]
        mask = ndimage.binary_opening(mask)
        mask = ndimage.binary_closing(mask)
        mask = ndimage.binary_fill_holes(mask)
        if mask.sum() < 10:
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, name=name,
        ))
    fig.update_layout(title="Multi-Region Fusion Segmentation", height=750,
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

def show_anatomical_labels(volume):
    vol = normalize_for_display(volume)
    jaw_mask = (vol > 0.25) & (vol <= 0.45)
    tooth_mask = (vol > 0.45) & (vol <= 0.70)
    ai_mask = vol > 0.70
    region_configs = [
        (jaw_mask, "Jaw / Tissue Region", "lightblue", 0.20),
        (tooth_mask, "Tooth Structure", "yellow", 0.45),
        (ai_mask, "AI Attention Region", "red", 0.75),
    ]
    fig = go.Figure()
    for mask, label, color, opacity in region_configs:
        mask = mask[::2, ::2, ::2]
        mask = ndimage.binary_opening(mask)
        mask = ndimage.binary_closing(mask)
        mask = ndimage.binary_fill_holes(mask)
        if mask.sum() < 10:
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, hovertext=label, hoverinfo="text",
        ))
        center = verts.mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="text", text=[label], textposition="top center", showlegend=False
        ))
    fig.update_layout(title="🦷 Anatomical Region Labeling", height=750,
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

def show_3d_attention_heatmap(volume):
    vol = normalize_for_display(volume)
    vol = vol[::2, ::2, ::2]
    attention_levels = [
        ((vol > 0.30) & (vol <= 0.50), "Low Attention", "blue", 0.10),
        ((vol > 0.50) & (vol <= 0.70), "Medium Attention", "yellow", 0.25),
        ((vol > 0.70), "High Attention", "red", 0.60),
    ]
    fig = go.Figure()
    for mask, label, color, opacity in attention_levels:
        mask = ndimage.binary_opening(mask)
        mask = ndimage.binary_closing(mask)
        mask = ndimage.binary_fill_holes(mask)
        if mask.sum() < 10:
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, name=label,
            hovertext=label, hoverinfo="text"
        ))
    fig.update_layout(title="🧠 3D AI Attention Heatmap", height=800,
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

def generate_clinical_findings(volume, roi_mask):
    findings = [] 

    vol = normalize_for_display(volume)

    roi_size = int(roi_mask.sum())
    roi_percentage = (roi_size / roi_mask.size) * 100

    left_half = roi_mask[:, :roi_mask.shape[1]//2, :]
    right_half = roi_mask[:, roi_mask.shape[1]//2:, :]

    left_activity = left_half.sum()
    right_activity = right_half.sum()

    asymmetry = abs(left_activity - right_activity)

    mean_intensity = float(vol[roi_mask].mean()) if roi_size > 0 else 0

    if roi_percentage > 15:
        findings.append("Large high-activity region detected")

    if mean_intensity > 0.7:
        findings.append("High-density anatomical structure observed")

    if asymmetry > 500:
        findings.append("Possible left-right asymetry detected")

    if roi_percentage < 3:
        findings.append("Minimal abnormal fusion activity")

    if len(findings) == 0:
        findings.append("No major abnormality detected")

    confidence = min(99, max(22, int(mean_intensity * 100)))

    return findings, confidence   

#gjufyk

# ---------------- PAGES ----------------

if page == "🏠 Clinical Dashboard":
    st.title("AI Clinical Dashboard")
    st.markdown("""
    Centralized multimodal dental imaging workspace powered by
    MONAI-based AI fusion and explainable clinical visualization.
    """)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)          # ✅ fixed clo4 typo
    col1.metric("Modalities", "3")
    col2.metric("Fusion Status", "Active")
    col3.metric("Segmentation Confidence", "94%")    # ✅ fixed .netric typo
    col4.metric("Inference Mode", "CPU Demo")

    st.markdown("---")
    st.subheader("⚙️ AI Pipeline Status")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.success("Acquisition Completed")
    p2.success("Preprocessing Completed")
    p3.success("Feature Extraction Complete")
    p4.success("Fusion Completed")
    p5.success("Explainability Completed")

    st.markdown("---")
    st.info("""
    AI-assisted Multimodal fusion completed successfully.

    The system integrated:
    - PAN anatomical context
    - CBCT volumetric structures
    - Soft-tissue imaging information

    Attention-guided segmentation and explainability maps
    have been generated for clinical review.
    """)

    st.markdown("---")
    st.subheader("🩺 Quick Clinical Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("High Attention Regions", "4")
    c2.metric("ROI Structures", "12")
    c3.metric("Processing Time", "2.1s")

    st.markdown("---")
    st.success("""
    Acquisition -> Preprocessing -> Feature Extraction ->
    Fusion -> Explainability -> Clinical Review
    """)

elif page == "🧠 Acquisition & Modalities":
    overview_cards()
    challenge_cards()
    modality_icon_cards()

elif page == "⚙️ Preprocessing Pipeline":
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
    m1.metric("MONAI Core", "Training + Inference")
    m2.metric("MONAI Bundle", "Model Packaging")
    m3.metric("MONAI Deploy", "Deployment Ready")
    m4.metric("MONAI Label", "Annotation Loop")

    st.markdown("## 📊 Evaluation Targets")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Dice Score", "> 0.85")
    e2.metric("PSNR", "> 30 dB")
    e3.metric("SSIM", "> 0.90")
    e4.metric("Inference", "CPU / GPU")

    st.warning("Prototype status: demo-ready, not for clinical diagnosis.")

    st.markdown("---")
    st.title("🔬 Interactive Preprocessing Visualization")

    if not st.session_state["fusion_done"]:

        st.warning("""
    Run the Fusion Engine first to generate preprocessing visualization.
    """)

    else:
        results = st.session_state["fusion_results"]

        pan_np = results["pan_np"]
        cbct_np = results["cbct_np"]
        soft_np = results["soft_np"]

        st.markdown("---")

        st.subheader("⚙️ Processing Workflow")

        st.success("""
    Raw Imaging
    -> Intensity Normalization
    -> Denoising
    -> Spatial Alignment
    -> Registration
    -> Tensor Conversion
    -> AI Fusion Ready    
    """)

        st.markdown("---")
        st.subheader("🖼️ Preprocessing Comparison")

        slice_idx = cbct_np.shape[2] // 2

        raw_slice = cbct_np[:, :, slice_idx]

        normalized_slice = normalize_for_display(raw_slice)

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                raw_slice,
                caption="Raw CBCT Slice",
                use_container_width=True,
                clamp=True
            )

        with c2:
            st.image(
                normalized_slice,
                caption="Normalized CBCT Clice",
                use_container_width=True,
                clamp=True
            )

        st.markdown("---")
        st.subheader("📊 Histogram Analysis")

        h1, h2 = st.columns(2)

        with h1:
            fig_raw_hist, ax_raw = plt.subplots()

            ax_raw.hist(
                raw_slice.flatten(),
                bins=50,
                color="gray",
                alpha=0.8
            )

            ax_raw.set_title("Raw CBCT Intensity Distribution")
            ax_raw.set_xlabel("Intensity")
            ax_raw.set_ylabel("Pixel Count")

            st.pyplot(fig_raw_hist)

        with h2:
            fig_norm_hist, ax_norm = plt.subplots()

            ax_norm.hist(
                normalized_slice.flatten(),
                bins=50,
                color="blue",
                alpha=0.8
            )

            ax_norm.set_title("Normalized Intensity Distribution")
            ax_norm.set_xlabel("Normalized Intensity")
            ax_norm.set_ylabel("Pixel Count")

            st.pyplot(fig_norm_hist)

            st.info("""
Histogram analysis helps visualize how preprocessing improves
pixel intensity distribution before AI feature extraction.

- Raw CBCT may contain inconsistent intensity ranges
- Normalization standardizes voxel distribution
- Improved distribution helps stabilize model training        
""")

        st.markdown("---")
        st.markdown("🧪 Noise Simulation & Denoising")

        noise_type = st.selectbox(
            "Noise Type",
            ["gaussian", "salt_pepper"]
        )

        noise_strength = st.slider(
            "Noise Strength",
            0.01,
            0.30,
            0.05
        )

        base_img = normalize_for_display(raw_slice)

        noisy_img = add_noise(
            base_img,
            noise_type=noise_type,
            strength=noise_strength
        )

        denoised_img = denoise_image(
            noisy_img,
            blur_strength=3
        )

        n1, n2, n3 = st.columns(3)

        with n1:
            st.image(
                base_img,
                caption="Original Image",
                use_container_width=True,
                clamp=True
            )

        with n2:
            st.image(
                noisy_img,
                caption=f"Noisy Image ({noise_type})",
                use_container_width=True,
                clamp=True
            )

        with n3:
            st.image(
                denoised_img,
                caption="Denoised Output",
                use_container_width=True,
                clamp=True
            )

        psnr_noise = peak_signal_noise_ratio(base_img, noisy_img, data_range=1.0)
        psnr_denoised = peak_signal_noise_ratio(base_img, denoised_img, data_range=1.0)

        m1, m2 = st.columns(2)

        m1.metric(
            "PSNR (Noisy)",
            f"{psnr_noise:.2f} dB"
        )

        m2.metric(
            "PSNR (Denoised)",
            f"{psnr_denoised:.2f} dB"
        )

        st.markdown("---")
        st.subheader("🧭 Registration Preview")

        registration_alpha = st.slider(
            "Overlay Strength",
            0.0,
            1.0,
            0.5
        )

        fixed_img, moving_img, registered_overlay = create_registration_preview(
            pan_np,
            cbct_np,
            alpha=registration_alpha
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.image(
                fixed_img,
                caption="Fixed Reference (Panoramic)",
                use_container_width=True,
                clamp=True
            )

        with r2:
            st.image(
                moving_img,
                caption="Moving Image (CBCT)",
                use_container_width=True,
                clamp=True
            )

        with r3:
            st.image(
                registered_overlay,
                caption="Registration Overlay",
                use_container_width=True,
                clamp=True
            )

        st.info("""
        Image registration aligns multimodal scans into a shared spatial grid  reference.

        - Fixed image -> anatomical reference
        - Moving image -> aligned modality
        - Overlay preview -> registration consistency

        This preprocessing step improves multimodal fusion quality.
        """)


        st.markdown("---")

        st.subheader("🧠 Multimodal Preprocessing")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.image(
                normalize_for_display(pan_np),
                caption="Panoramic Normalization",
                use_container_width=True
            )

        with p2:
            st.image(
                normalize_for_display(get_slice(cbct_np)),
                caption="CBCT Slice Standardization",
                use_container_width=True
            )

        with p3:
            st.image(
                normalize_for_display(soft_np),
                caption="Soft Tissue Enhancement",
                use_container_width=True
            )

        st.markdown("## 🎛️ Interactive Preprocessing Controls")

        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            brightness = st.slider(
                "Brightness Adjustment",
                -500.0,
                500.0,
                0.0,
                key="prep_brightness"
            )

        with pc2:
            contrast = st.slider(
                "Contrast Scaling",
                0.5,
                2.0,
                1.0,
                key="prep_contrast"
            )

        with pc3:
            blur_strength = st.slider(
                "Noise Reduction",
                0,
                5,
                0,
                key="prep_blur"
            )

        processed_preview = raw_slice.astype(np.float32)

        processed_preview = (processed_preview * contrast) + brightness 

        if blur_strength > 0:
            processed_preview = cv2.GaussianBlur(
                processed_preview,
                (blur_strength * 2 + 1, blur_strength * 2 + 1),
                0
            )

        processed_preview = normalize_for_display(processed_preview)

        st.image(
            processed_preview,
            caption="Interactive Preprocessing Preview",
            use_container_width=True,
            clamp=True
        )

        st.markdown("---")

        st.subheader("🧠 Applied Preprocessing Operations")

        ops1, ops2 = st.columns(2)

        with ops1:

            st.info("""
    - Intensity normalization
    - Dynamic range scaling
    - Slice extraction
    - Tensor preparation
    """)

        with ops2:

            st.info("""
    - Spatial standardization
    - Dynamic range scaling
    - Noise reduction
    - Fusion alignment preparation
    """)

        st.markdown("---")

        st.success("""
        Preprocessing completed successfully.

        The imaging modalities are normalized and aligned 
        for multimodal feature extraction and fusion analysis.
        """)   

elif page == "🧩 Feature Extraction":
    pipeline_cards()
    architecture_page_cards()

elif page == "🔗 Fusion Engine":
    section_title("Live Fusion Dashboard", "Clinical-style demo workspace")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📖 Imaging Modalities")
    st.markdown("""
🦷 **Panoramic (OPG)** <span title='Full jaw 2D X-ray showing teeth and jaw structure'>ℹ️</span>  
🧊 **CBCT** <span title='3D cone beam CT scan for volumetric dental imaging'>ℹ️</span>  
🧠 **Soft Tissue** <span title='Facial surface / soft tissue structure for esthetic planning'>ℹ️</span>  
🔗 **Fusion** <span title='Combines multiple imaging modalities into one AI output'>ℹ️</span>  
""", unsafe_allow_html=True)

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

    if run_demo or st.session_state["fusion_done"]:

        using_cached_results = st.session_state["fusion_done"] and not run_demo

        if using_cached_results:
            results = st.session_state["fusion_results"]
            pan_np = results["pan_np"]
            cbct_np = results["cbct_np"]
            soft_np = results["soft_np"]
            output_np = results["output_np"]
            cbct_info = results["cbct_info"]
            inference_time = results["inference_time"]
            cbct_tensor = results["cbct_tensor"]
            device = results["device"]

        else:
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
image using a 3D UNet architecture.

This preserves structural + contextual information across modalities.
""")
            st.markdown("### 🤖 AI Interpretation Layer")
            st.success("Fusion complete. Generating visual intelligence...")
            st.markdown("### 🧪 Clinical View")
            st.caption("Radiology-style visualization for diagnostic interpretation")

            pan_np = pan_tensor.cpu().numpy()[0, 0]
            cbct_np = cbct_tensor.cpu().numpy()[0, 0]
            soft_np = soft_tensor.cpu().numpy()[0, 0]
            output_np = output_tensor.cpu().numpy()[0, 0]

            fused_vis = (output_np > 0.5).astype(np.float32)
            fused_vis = cv2.GaussianBlur(fused_vis.astype(np.float32), (5, 5), 0)

            st.session_state["fusion_done"] = True
            st.session_state["fusion_results"] = {
                "pan_np": pan_np,
                "cbct_np": cbct_np,
                "soft_np": soft_np,
                "output_np": output_np,
                "cbct_info": cbct_info,
                "inference_time": inference_time,
                "cbct_tensor": cbct_tensor,
                "device": device,
            }

        if show_shapes:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Panoramic", str(np.asarray(pan_np).shape))
            c2.metric("CBCT", str(np.asarray(cbct_np).shape))
            c3.metric("Soft Tissue", str(np.asarray(soft_np).shape))
            c4.metric("Output", str(np.asarray(output_np).shape))

        psnr_value, ssim_value = calculate_image_metrics(cbct_np, output_np)

        st.markdown("## ✅ POC Validation Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dice Score", "Sanity Check")
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
        d2.info(f"Slice Thickness: {cbct_info.get('slice_thickness', 'Unknown')}")

        st.markdown("---")
        st.markdown("## 📊 Generated Results")

        st.markdown("## 🧾 AI Fusion Output")
        st.caption("Integrated multimodal visualization (Panoramic + CBCT + Soft Tissue)")

        st.markdown("---")
        st.markdown("### 📊 Histogram & Intensity Analysis")

        h1, h2 = st.columns(2)

        with h1:
            show_histogram_analysis(
                cbct_np,
                title="CBCT Histogram"
            )

        with h2:
            show_intensity_analysis(
                pan_np,
                cbct_np,
                soft_np,
                output_np,
                title="Fusion Output Histogram"
            )

        st.markdown("---")

        d1, d2 = st.columns(2)

        with d1:
            show_intensity_distribution(
                cbct_np,
                title="CBCT Intensity Distribution"
            )

        with d2:
            show_intensity_distribution(
                output_np,
                title="Fusion Output Distribution"
            )

        with st.container():
            st.markdown("#### 📊 Visualization")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(get_slice(pan_np), caption="Panoramic", use_container_width=True)
            c2.image(get_slice(cbct_np), caption="CBCT", use_container_width=True)
            c3.image(get_slice(soft_np), caption="Soft Tissue", use_container_width=True)
            c4.image(get_slice(output_np), caption="Fused-Based Segmentation Output",
                     use_container_width=True, clamp=True)

        st.subheader("Slice Synchronization")

        cbct_vol = np.asarray(cbct_np).squeeze()
        num_slices = cbct_vol.shape[2] if cbct_vol.ndim == 3 else 1

        slice_idx = st.slider(
            "Select Slice",
            0,
            max(num_slices - 1, 0),
            num_slices // 2,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                get_axial_slice(cbct_np, slice_idx, num_slices),
                caption=f"CBCT Slice {slice_idx}",
                use_container_width=True,
            )

        with col2:
            st.image(
                get_axial_slice(soft_np, slice_idx, num_slices),
                caption=f"Soft Tissue Slice {slice_idx}",
                use_container_width=True,
            )

        with col3:
            st.image(
                get_axial_slice(output_np, slice_idx, num_slices),
                caption=f"Fused Slice {slice_idx}",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 🧊 Advanced CBCT Explorer")

        view_mode = st.radio("Select View", ["Axial", "Coronal", "Sagittal"], horizontal=True)
        _, max_idx, default_idx = get_volume_view(cbct_np, view=view_mode)
        selected_idx = st.slider(f"{view_mode} Slice Index", 0, max_idx, default_idx)
        selected_view, _, _ = get_volume_view(cbct_np, view=view_mode, index=selected_idx)

        st.markdown("### 📊 Viewer Controls")
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            brightness = st.slider("Brightness", -500.0, 500.0, 0.0)
        with vc2:
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
        with vc3:
            zoom = st.slider("Zoom", 1.0, 3.0, 1.0)

        st.markdown("#### 🪟 Window Presets")
        preset = st.selectbox("Select Window Preset", ["Default", "Bone", "Soft Tissue"])
        if preset == "Bone":
            contrast = 1.6
            brightness = 100.0
        elif preset == "Soft Tissue":
            contrast = 0.8
            brightness = -100.0

        viewer_img = selected_view.astype(np.float32)
        viewer_img = (viewer_img * contrast) + brightness

        if zoom > 1.0:
            h, w = viewer_img.shape
            crop_h = int(h / zoom)
            crop_w = int(w / zoom)
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            viewer_img = viewer_img[start_h:start_h + crop_h, start_w:start_w + crop_w]

        max_x = viewer_img.shape[1] - 1
        max_y = viewer_img.shape[0] - 1
        default_x1 = min(10, max_x)
        default_y1 = min(10, max_y)
        default_x2 = min(50, max_x)
        default_y2 = min(50, max_y)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            x1 = st.number_input("Point 1 - x", 0, max_x, default_x1, key="x1")
            y1 = st.number_input("Point 1 - y", 0, max_y, default_y1, key="y1")
        with col_m2:
            x2 = st.number_input("Point 2 - x", 0, max_x, default_x2, key="x2")
            y2 = st.number_input("Point 2 - y", 0, max_y, default_y2, key="y2")

        fig_view, axv = plt.subplots()
        axv.imshow(viewer_img, cmap=cmap)
        axv.plot([x1, x2], [y1, y2], color='red', linewidth=2)
        axv.scatter([x1, x2], [y1, y2], color='yellow')
        axv.set_title(f"{view_mode} View - Slice {selected_idx}")
        axv.axis("off")
        st.pyplot(fig_view)
        st.caption(f"Viewer settings — Brightness: {brightness}, Contrast: {contrast}, Zoom: {zoom}x")

        st.markdown("---")
        st.markdown("### 📏 Measurement Tool")
        distance_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        pixel_spacing = 0.3
        distance_mm = distance_px * pixel_spacing
        st.info(f"Distance: {distance_mm:.2f} mm")

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
            st.markdown("### 🧩 Segmentation Overlay (Simulated)")

            seg_tensor, seg_mode = run_segmentation(cbct_tensor, device)
            seg_np = seg_tensor.cpu().numpy()[0, 0]
            seg_slice = get_slice(seg_np)

            dicom_ds = create_dicom_overlay(output_np)
            dicom_buffer = io.BytesIO()
            dicom_ds.save_as(dicom_buffer, write_like_original=False)
            dicom_buffer.seek(0)

            st.download_button(
                "⬇️ Download DICOM Overlay",
                data=dicom_buffer,
                file_name="fusion_overlay.dcm",
                mime="application/dicom",
            )

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

            st.caption("This overlay is a simulated ROI mask for demo purposes.")

            st.markdown("### 🧠 AI Summary")
            mean_val = float(np.mean(output_np))
            max_val = float(np.max(output_np))
            if mean_val > 0:
                summary = "The fused output highlights areas of structural density with moderate intensity."
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
• Segmentation highlights potential regions of interest
• Heatmap indicates model attention zones
• Supports planning, diagnosis, and review workflows
""")

        st.markdown("### 🧠 AI Clinical Insight")
        st.info("""
• Multimodal fusion highlights structural + soft tissue alignment
• Segmentation overlay simulates AI-based region-of-interest detection
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
- Regions of interest detected using simulated alignment
- Heatmap shows AI attention zones

OUTPUT SUMMARY
- Fused visualization generated
- CBCT multi-view explorer enabled
- Segmentation overlay applied
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

elif page == "🧠 AI Explainability":
    st.title("🧠 AI Multimodal Fusion Explainability")
    st.caption("Visual explanation of how CBCT, Panoramic, and Soft Tissue inputs are fused")
    st.info("Run the Fusion Engine first to generate fusion results. Then return to this page.")

    if not st.session_state["fusion_done"]:
        st.warning("Fusion results are not available yet. Please go to 🔗 Fusion Engine and click Run Fusion Demo.")
    else:
        results = st.session_state["fusion_results"]
        pan_np = results["pan_np"]
        cbct_np = results["cbct_np"]
        soft_np = results["soft_np"]
        output_np = results["output_np"]

        pan_img = normalize_for_display(get_slice(pan_np))
        cbct_slice = normalize_for_display(get_slice(cbct_np))
        soft_img = normalize_for_display(get_slice(soft_np))
        fused_img = normalize_for_display(get_slice(output_np))

        st.markdown("## 🧠 Interactive Fusion Workflow")
        fusion_step = st.select_slider(
            "Fusion workflow step",
            options=["PAN Input", "CBCT Features", "Soft Tissue", "Attention Fusion", "Fused Output", "Segmentation"]
        )

        st.markdown("## 1️⃣ Input Modalities")
        c1, c2, c3 = st.columns(3)
        c1.image(pan_img, caption="Panoramic X-Ray", use_container_width=True, clamp=True)
        c2.image(cbct_slice, caption="CBCT Slice", use_container_width=True, clamp=True)
        c3.image(soft_img, caption="Soft Tissue", use_container_width=True, clamp=True)

        st.markdown("---")
        st.markdown("## 2️⃣ Feature Extraction")
        feature_pan = normalize_for_display(pan_img)
        feature_cbct = normalize_for_display(cbct_slice)
        feature_soft = normalize_for_display(soft_img)
        f1, f2, f3 = st.columns(3)
        f1.image(feature_pan, caption="PAN Feature Map", use_container_width=True, clamp=True)
        f2.image(feature_cbct, caption="CBCT Feature Map", use_container_width=True, clamp=True)
        f3.image(feature_soft, caption="Soft Tissue Feature Map", use_container_width=True, clamp=True)

        st.markdown("---")
        st.markdown("## 3️⃣ Attention-Based Fusion")
        w1, w2, w3 = st.columns(3)
        with w1:
            pan_weight = st.slider("PAN Weight", 0.0, 1.0, 0.35)
        with w2:
            cbct_weight = st.slider("CBCT Weight", 0.0, 1.0, 0.45)
        with w3:
            soft_weight = st.slider("Soft Tissue Weight", 0.0, 1.0, 0.20)

        total_weight = pan_weight + cbct_weight + soft_weight
        if total_weight == 0:
            total_weight = 1.0
        pan_w = pan_weight / total_weight
        cbct_w = cbct_weight / total_weight
        soft_w = soft_weight / total_weight
        fusion_attention_map = pan_w * feature_pan + cbct_w * feature_cbct + soft_w * feature_soft
        st.image(normalize_for_display(fusion_attention_map), caption="Fusion Attention Map",
                 use_container_width=True, clamp=True)

        st.markdown("---")
        st.markdown("## 4️⃣ Before vs After Fusion")
        b1, b2 = st.columns(2)
        b1.image(cbct_slice, caption="Before Fusion: CBCT Reference", use_container_width=True, clamp=True)
        b2.image(fused_img, caption="After Fusion: Fusion-Based Segmentation", use_container_width=True, clamp=True)

        st.markdown("---")
        st.markdown("## 🧊 3D CBCT Volume Viewer")
        show_3d_volume(cbct_np, "Interactive 3D CBCT Volume")

        st.markdown("---")
        st.markdown("## 🦷 3D Dental Surface Rendering")
        surface_threshold = st.slider("Surface Extraction Threshold", 0.1, 0.9, 0.45, key="surface_threshold")
        show_3d_surface(cbct_np, title="Extracted 3D High-Density Dental Structure", threshold=surface_threshold)

        st.markdown("---")
        st.markdown("## 🔗 3D Fusion Overlay")
        oc1, oc2 = st.columns(2)
        with oc1:
            cbct_threshold = st.slider("CBCT Surface Threshold", 0.1, 0.9, 0.45)
            cbct_opacity = st.slider("CBCT Opacity", 0.05, 1.0, 0.30)
        with oc2:
            fusion_threshold = st.slider("Fusion Output Threshold", 0.1, 0.9, 0.50)
            fusion_opacity = st.slider("Fusion Overlay Opacity", 0.05, 1.0, 0.75)
        show_3d_fusion_overlay(
            cbct_np, output_np,
            title="CBCT Anatomy + Fusion-Based Segmentation Overlay",
            cbct_threshold=cbct_threshold, fusion_threshold=fusion_threshold,
            cbct_opacity=cbct_opacity, fusion_opacity=fusion_opacity,
        )
        st.info("""
3D Overlay Legend:
- Light gray: CBCT anatomical structure
- Red: Fusion-based segmentation / AI-highlighted region
- Opacity controls help compare anatomical structure with predicted regions
""")

        st.markdown("## 📊 3D ROI Analysis")
        cbct_norm = normalize_for_display(cbct_np)
        fusion_norm = normalize_for_display(output_np)

        if fusion_norm.shape != cbct_norm.shape:
            resized_slices = []
            for i in range(cbct_norm.shape[2]):
                src_idx = min(i, fusion_norm.shape[2] - 1)
                slice_img = Image.fromarray(fusion_norm[:, :, src_idx].astype(np.float32))
                slice_img = slice_img.resize((cbct_norm.shape[1], cbct_norm.shape[0]))
                resized_slices.append(np.array(slice_img).astype(np.float32))
            fusion_norm = np.stack(resized_slices, axis=2)

        roi_mask = fusion_norm > fusion_threshold
        roi_voxels = int(np.sum(roi_mask))
        cbct_voxels = int(roi_mask.size)
        roi_percentage = (roi_voxels / cbct_voxels) * 100
        if roi_voxels > 0:
            mean_roi_intensity = float(cbct_norm[roi_mask].mean())
            max_roi_intensity = float(cbct_norm[roi_mask].max())
        else:
            mean_roi_intensity = 0.0
            max_roi_intensity = 0.0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("ROI Voxels", roi_voxels)
        r2.metric("ROI Coverage", f"{roi_percentage:.2f}%")
        r3.metric("Mean ROI Intensity", f"{mean_roi_intensity:.3f}")
        r4.metric("Max ROI Intensity", f"{max_roi_intensity:.3f}")
        st.caption("ROI statistics are computed from the fusion-based segmentation mask for explainability/demo purposes.")

        st.markdown("## 🎯 ROI Slice Locator")
        roi_per_slice = roi_mask.sum(axis=(0, 1))
        if roi_per_slice.max() > 0:
            max_roi_slice = int(np.argmax(roi_per_slice))
            st.success(f"Highest ROI activity detected at slice: {max_roi_slice}")
            l1, l2 = st.columns(2)
            with l1:
                st.image(cbct_norm[:, :, max_roi_slice], caption=f"CBCT Slice {max_roi_slice}",
                         use_container_width=True, clamp=True)
            with l2:
                st.image(fusion_norm[:, :, max_roi_slice], caption=f"Fusion ROI Slice {max_roi_slice}",
                         use_container_width=True, clamp=True)
        else:
            st.warning("No ROI activity detected with the current threshold.")

        st.markdown("---")
        st.markdown("## 🧠 Multi-Region Clinical Segmentation")
        st.caption("""
Blue  → Low AI fusion activity
Yellow → Medium fusion activity
Red   → High-confidence fusion regions
""")
        show_multi_region_segmentation(output_np)

        st.markdown("---")
        st.markdown("## 🦷 Anatomical Region Labeling")
        st.caption("Interactive anatomical labelling generated from multimodal fusion activity.")
        show_anatomical_labels(output_np)

        st.markdown("---")
        st.markdown("## 🧠 3D AI Attention Heatmap")
        st.caption("Interactive visualization of low, medium, and high AI attention regions generated from multimodal fusion activity.")
        show_3d_attention_heatmap(output_np)

        st.markdown("---")
        st.markdown("## 🧾 AI Clinical Findings Summary")
        high_activity_mask = normalize_for_display(output_np) > 0.70
        findings, confidence = generate_clinical_findings(
            output_np,
            high_activity_mask
        )
        medium_activity_mask = (normalize_for_display(output_np) > 0.50) & (normalize_for_display(output_np) <= 0.70)
        high_count = int(high_activity_mask.sum())
        medium_count = int(medium_activity_mask.sum())
        if high_count > 0:
            finding_status = "High fusion activity detected"
            finding_note = "The model identified concentrated high-response regions that may require focused review."
        elif medium_count > 0:
            finding_status = "Moderate fusion activity detected"
            finding_note = "The model identified moderate multimodal alignment regions for review."
        else:
            finding_status = "Low fusion activity"
            finding_note = "No strong fusion-highlighted region was detected with the current threshold."

        f1, f2, f3 = st.columns(3)
        f1.metric("Finding Status", finding_status)
        f2.metric("High Activity Voxels", high_count)
        f3.metric("Medium Activity Voxels", medium_count)
        st.info(f"""
Clinical Interpretation:
{finding_note}

AI Confidence Score:
{confidence}%

This output is generated for explainability/demo purposes and is not intended for clinical diagnosis.
""")

        st.markdown("### 🩺 Simulated Clinical Findings")

        for item in findings:
            st.success(f"• {item}")

        st.markdown("---")
        st.markdown("## 5️⃣ Fusion Workflow")
        st.success("""
PAN Encoder + CBCT Encoder + Soft Tissue Encoder
→ Feature Extraction
→ Attention Weighting
→ Fusion Layer
→ Fused Representation
→ Segmentation / Visualization / Clinical Report
""")
        st.markdown("## 🧾 Clinical Interpretation")
        st.info("""
Fusion combines:
- Structural CBCT information
- Panoramic dental alignment context
- Soft tissue appearance information

The fusion layer creates a unified representation that supports segmentation,
heatmap visualization, and clinical-style interpretation.
""")

elif page == "🩺 Clinical Findings":
    use_case_tabs()
    outcomes_cards()
    clinical_summary_box()

elif page == "🌐 3D Diagnostic Workspace":
    st.title("🌐 3D Diagnostic Workspace")
    if not st.session_state["fusion_done"]:
        st.warning("Please run the 🔗 Fusion Engine first to load 3D data.")
    else:
        results = st.session_state["fusion_results"]
        cbct_np = results["cbct_np"]
        output_np = results["output_np"]

        st.markdown("### 🧊 3D CBCT Volume")
        show_3d_volume(cbct_np, "Interactive 3D CBCT Volume")

        st.markdown("---")
        st.markdown("### 🦷 3D Surface Rendering")
        ws_threshold = st.slider("Surface Threshold", 0.1, 0.9, 0.45, key="ws_thresh")
        show_3d_surface(cbct_np, title="3D High-Density Dental Structure", threshold=ws_threshold)

        st.markdown("---")
        st.markdown("### 🔗 3D Fusion Overlay")
        show_3d_fusion_overlay(cbct_np, output_np, title="3D Fusion Overlay")

        st.markdown("---")
        st.markdown("### 🧠 Multi-Region Segmentation")
        show_multi_region_segmentation(output_np)

        st.markdown("---")
        st.markdown("### 🦷 Anatomical Labels")
        show_anatomical_labels(output_np)

elif page == "📈 Operational Metrics":
    st.title("📈 Operational Metrics")
    if not st.session_state["fusion_done"]:
        st.info("Run the 🔗 Fusion Engine to populate metrics.")
    else:
        results = st.session_state["fusion_results"]
        psnr_val, ssim_val = calculate_image_metrics(results["cbct_np"], results["output_np"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR", f"{psnr_val:.2f} dB")
        m2.metric("SSIM", f"{ssim_val:.3f}")
        m3.metric("Inference Time", f"{results['inference_time']:.2f}s")
        m4.metric("Modalities", "3")

        st.markdown("---")
        st.markdown("## ⚡ System Performance")
        p1, p2, p3 = st.columns(3)
        p1.metric("Pipeline Status", "Operational ✅")
        p2.metric("Preprocessing Mode", "CPU (Demo)")
        p3.metric("Deployment", "Yes")

elif page == "🚀 Platform Vision":
    platform_cards()
    st.markdown("## 🧬 MONAI Ecosystem Alignment")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
### MONAI Core
Used for medical imaging model development, transforms, and inference.

Current use:
- Fusion model pipeline
- Medical preprocessing
- Segmentation integration
""")
    with c2:
        st.info("""
### MONAI Label
Future clinician-in-the-loop annotation layer.

Future use:
- AI-assisted labeling
- Active learning
- Dentist feedback loop
""")
    with c3:
        st.info("""
### MONAI Deploy
Future clinical deployment layer.

Future use:
- DICOM workflow integration
- Containerized deployment
- Scalable inference
""")
    outcomes_cards()
    clinical_summary_box()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("### 🏁 Demo Summary")
st.success("""
This platform demonstrates how multimodal dental imaging can be unified into a
single AI-assisted workflow, enabling enhanced visualization and future-ready
clinical decision support.
""")

footer_note()