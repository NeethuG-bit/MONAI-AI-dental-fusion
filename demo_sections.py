import streamlit as st

def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

def hero_section():
    st.markdown("# 🦷 Dental AI Fusion Platform")
    st.write(
        "Premium proof-of-concept demo for multimodal dental imaging using "
        "panoramic X-ray, CBCT, and soft tissue fusion."
    )

def kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modalities", "3")
    c2.metric("Fusion", "Feature-Level")
    c3.metric("Mode", "Clinical POC")
    c4.metric("Inference", "CPU Demo")

def workflow_banner():
    st.info(
        "Workflow: Acquisition → Preprocessing → Feature Extraction → Fusion → Visualization"
    )

def overview_cards():
    section_title("Platform Overview", "Multimodal AI-driven dental imaging system")

    st.write("""
This platform demonstrates how multiple dental imaging modalities can be combined
into a unified AI-powered workflow.

It integrates:
- 2D panoramic X-rays
- 3D CBCT volumetric scans
- Soft tissue / facial surface data

The goal is to enhance visualization, improve interpretability, and support
future-ready clinical workflows using AI-assisted fusion.
""")

    st.markdown("### Key Capabilities")

    col1, col2, col3 = st.columns(3)

    col1.info("🧠 Multimodal Fusion\nCombines multiple imaging sources into one view")
    col2.info("📊 Visualization\nClear representation of fused outputs")
    col3.info("⚡ Real-time Demo\nInteractive clinical-style interface")

def challenge_cards():
    section_title("Clinical Challenges")
    c1, c2, c3 = st.columns(3)
    c1.warning("Fragmented imaging review")
    c2.warning("Limited soft tissue context")
    c3.warning("Time-intensive manual interpretation")

def modality_icon_cards():
    section_title("Core Input Modalities")
    c1, c2, c3 = st.columns(3)
    c1.success("🩻 Panoramic X-ray")
    c2.success("🧊 CBCT Volume")
    c3.success("🫧 Soft Tissue")

def pipeline_cards():
    section_title("Pipeline Architecture")
    steps = [
        "1. Multimodal Acquisition",
        "2. Registration / Normalization",
        "3. Feature Extraction",
        "4. Fusion Engine",
        "5. Reconstruction / Visualization",
        "6. Clinical Dashboard",
    ]
    for step in steps:
        st.write(step)

def upload_console():
    section_title("Clinical Input Console")
    c1, c2, c3 = st.columns(3)
    with c1:
        pan_file = st.file_uploader("Upload panoramic image", type=["png", "jpg", "jpeg"], key="pan_upload")
    with c2:
        cbct_file = st.file_uploader("Upload CBCT preview image", type=["png", "jpg", "jpeg"], key="cbct_upload")
    with c3:
        soft_file = st.file_uploader("Upload soft tissue image", type=["png", "jpg", "jpeg"], key="soft_upload")
    return pan_file, cbct_file, soft_file

def use_case_tabs():
    section_title(
        "Clinical Use Cases",
        "Key applications of the multimodal dental AI fusion platform."
    )

    tabs = st.tabs(["Orthodontics", "Implants", "TMJ", "Surgery"])

    with tabs[0]:
        st.markdown("### Orthodontics")
        st.write("""
Orthodontic planning involves evaluating tooth alignment, jaw relationships,
eruption patterns, and overall craniofacial structure.

With multimodal fusion, the platform can combine:
- panoramic imaging for overall dentition review
- CBCT for 3D spatial relationships
- soft tissue views for facial profile assessment

This supports:
- treatment planning
- impacted tooth analysis
- arch development review
- patient communication with clearer visual context
""")

    with tabs[1]:
        st.markdown("### Implants")
        st.write("""
Implant planning requires accurate understanding of bone availability,
tooth position, and surrounding anatomical structures.

The fusion platform helps by integrating:
- panoramic context for full-jaw overview
- CBCT for bone volume and depth assessment
- soft tissue information for esthetic planning

This supports:
- implant site planning
- bone quality visualization
- proximity assessment to critical structures
- prosthetic and surgical workflow preparation
""")

    with tabs[2]:
        st.markdown("### TMJ")
        st.write("""
TMJ analysis focuses on the temporomandibular joint, surrounding bone structures,
jaw alignment, and possible functional or structural abnormalities.

The fused platform can support TMJ-oriented workflows by providing:
- structural review through panoramic imaging
- volumetric detail from CBCT
- external soft tissue context for facial symmetry and posture review

This can be extended for:
- jaw asymmetry evaluation
- condylar positioning analysis
- structural screening support
- advanced multimodal diagnostic workflows
""")

    with tabs[3]:
        st.markdown("### Surgery")
        st.write("""
Surgical planning in dental and maxillofacial care requires accurate visualization
of anatomy before intervention.

The fusion platform supports this by combining:
- panoramic overview of dentition and jaw structure
- CBCT-based 3D anatomical understanding
- soft tissue surface context for external planning perspective

This is useful for:
- pre-surgical simulation
- impacted tooth surgery planning
- implant-related procedures
- guided visualization for patient explanation
- future roadmap integration with surgical navigation workflows
""")

def platform_cards():
    section_title("Platform Capabilities", "What this system can do")

    st.write("""
The platform is designed as a proof-of-concept for AI-assisted dental imaging
systems with interactive capabilities.
""")

    st.markdown("### Core Features")

    col1, col2 = st.columns(2)

    col1.write("""
- Multimodal data handling
- Interactive UI dashboard
- Real-time model inference (CPU demo)
- Image upload + preview
""")

    col2.write("""
- Visualization of fused outputs
- Slice exploration (CBCT)
- Simulated AI heatmaps
- Downloadable outputs
""")

    st.markdown("### Future Scope")

    st.write("""
- Integration with real clinical datasets
- GPU-based inference
- Advanced 3D visualization
- Automated report generation
""")

def architecture_page_cards():
    section_title("System Architecture", "How the fusion pipeline works")

    st.write("""
The system follows a modular deep learning pipeline designed for multimodal
integration of dental imaging data.
""")

    st.markdown("### Pipeline Stages")

    st.write("""
1. **Acquisition**
   - Input from panoramic, CBCT, and soft tissue sources

2. **Preprocessing**
   - Normalization
   - Resizing
   - Modality alignment

3. **Feature Extraction**
   - Deep learning encoders extract modality-specific features

4. **Fusion Layer**
   - Features are combined into a unified representation

5. **Visualization**
   - Outputs are rendered for interpretation
""")

    st.markdown("### Technology Stack")

    st.write("""
- PyTorch (Model implementation)
- MONAI (Medical imaging framework)
- Streamlit (Interactive UI layer)
""")

def outcomes_cards():
    section_title("Expected Outcomes")
    c1, c2, c3, c4 = st.columns(4)
    c1.success("Planning Efficiency")
    c2.success("Better Visualization")
    c3.success("Treatment Support")
    c4.success("POC Readiness")

def clinical_summary_box():
    st.info(
        "This live app demonstrates the fusion experience and UI direction. "
        "Advanced registration, temporal modeling, and full deployment should "
        "be presented as roadmap layers unless fully implemented."
    )

def footer_note():
    st.caption(
        "Proof-of-concept note: synthetic demo data is used for live visualization."
    )