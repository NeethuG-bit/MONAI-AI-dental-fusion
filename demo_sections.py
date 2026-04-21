import streamlit as st

def section_title(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def hero_section():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-badge">Proof-of-Concept • Clinical Product Demo</div>
            <div class="hero-title">🦷 Dental AI Fusion Platform</div>
            <div class="hero-subtitle">
                Multimodal dental imaging demo for panoramic X-ray, CBCT, and soft tissue
                fusion. Built to present a product-oriented clinical workflow for diagnostics,
                visualization, and treatment planning support.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def overview_cards():
    section_title(
        "Executive Overview",
        "A product-style summary of the workflow, value proposition, and clinical fit."
    )
    cols = st.columns(3)
    items = [
        ("Unified Visualization", "Bring panoramic, CBCT, and soft tissue views into one guided review experience."),
        ("Clinical Workflow Support", "Reduce manual mental stitching across imaging modalities and improve treatment planning efficiency."),
        ("Deployment Story", "Position the solution as MONAI-based today with future edge-ready or clinical deployment potential."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def challenge_cards():
    section_title(
        "Clinical Challenges",
        "These directly mirror the user-facing problem framing expected in the POC."
    )
    cols = st.columns(3)
    items = [
        ("Fragmented Data Sources", "Separate review of panoramic, CBCT, and surface/soft tissue data slows interpretation."),
        ("Limited Soft Tissue Context", "Traditional imaging workflows emphasize hard tissue but may not sufficiently unify soft tissue context."),
        ("Time-Intensive Analysis", "Manual correlation across modalities depends heavily on clinician experience and time."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def modalities_cards():
    section_title(
        "Imaging Modalities",
        "The live demo uses synthetic sample data to represent these three input streams."
    )
    cols = st.columns(3)
    items = [
        ("2D Panoramic X-ray", "Provides broad contextual overview of dentition and maxillofacial structures."),
        ("3D CBCT Volume", "Captures spatial and volumetric hard tissue information for planning and structural interpretation."),
        ("Soft Tissue Surface", "Adds gingival and contour-aware context to support a more complete planning view."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def pipeline_cards():
    section_title(
        "Pipeline Architecture",
        "Presenting the workflow as a product pipeline is important for matching the PDF’s expected demo structure."
    )
    pipeline = [
        ("1. Acquisition", "Collect panoramic, CBCT, and soft tissue inputs."),
        ("2. Registration", "Prepare aligned multimodal representations for downstream analysis."),
        ("3. Normalization", "Standardize modalities for model ingestion."),
        ("4. Feature Extraction", "2D and 3D encoders extract modality-specific representations."),
        ("5. Feature-Level Fusion", "Fuse shared information across modalities into a single representation."),
        ("6. Reconstruction / Output", "Generate clinically reviewable fused output."),
        ("7. Clinical Review", "Present results in a guided decision-support interface."),
    ]
    for title, text in pipeline:
        st.markdown(
            f'<div class="pipeline-box"><strong>{title}</strong><br>{text}</div>',
            unsafe_allow_html=True,
        )

def upload_placeholders():
    section_title(
        "Clinical Input Console",
        "These upload controls make the demo feel product-ready even if the current backend uses synthetic data."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="upload-box">Panoramic Input</div>', unsafe_allow_html=True)
        st.file_uploader("Upload panoramic image", type=["png", "jpg", "jpeg"], key="pan_upload")
    with col2:
        st.markdown('<div class="upload-box">CBCT Volume Placeholder</div>', unsafe_allow_html=True)
        st.file_uploader("Upload CBCT slice/preview", type=["png", "jpg", "jpeg"], key="cbct_upload")
    with col3:
        st.markdown('<div class="upload-box">Soft Tissue Input</div>', unsafe_allow_html=True)
        st.file_uploader("Upload soft tissue image", type=["png", "jpg", "jpeg"], key="soft_upload")

def use_case_tabs():
    section_title(
        "Clinical Use Cases",
        "The PDF highlights these as primary application areas for the platform."
    )
    tabs = st.tabs([
        "Orthodontic Planning",
        "Implant Placement",
        "TMJ Analysis",
        "Surgical Simulation",
    ])

    with tabs[0]:
        st.markdown("Integrated hard and soft tissue context for tooth movement review, planning, and visual communication.")
    with tabs[1]:
        st.markdown("3D bone context plus surface awareness for implant planning and safer structural interpretation.")
    with tabs[2]:
        st.markdown("Position as an extensible architecture for TMJ and functional analysis workflows.")
    with tabs[3]:
        st.markdown("Use fused visualizations to support pre-operative simulation and planning discussions.")

def platform_cards():
    section_title(
        "Platform Layers",
        "A product-style explanation of how the system should be narrated during demo."
    )
    cols = st.columns(4)
    items = [
        ("MONAI Layer", "Medical imaging transforms, architecture support, and inference pipeline."),
        ("Fusion Layer", "2D/3D/soft tissue encoders combining into a unified representation."),
        ("Clinical UI Layer", "Guided interface for reviewing results rather than inspecting code or tensors."),
        ("Deployment Layer", "Cloud demo now; edge or clinic-ready positioning as future deployment path."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def outcomes_cards():
    section_title(
        "Expected Outcomes",
        "These cards help align the app with the PDF’s business and clinical language."
    )
    cols = st.columns(4)
    items = [
        ("Faster Planning", "Reduce time spent correlating imaging sources manually."),
        ("Improved Visualization", "Present more complete multimodal context in one place."),
        ("Clinical Support", "Frame the system as a decision-support tool for treatment planning."),
        ("Demo Credibility", "Make the POC feel closer to a deployable product experience."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def clinical_summary_box():
    st.markdown(
        """
        <div class="clinical-note">
        <strong>Clinical Summary</strong><br>
        This demo showcases a multimodal fusion experience rather than a validated clinical device.
        Use it to demonstrate product direction, workflow integration, and UI/UX readiness while
        clearly positioning advanced elements such as real registration, temporal modeling, and
        deployment optimization as roadmap or architecture components.
        </div>
        """,
        unsafe_allow_html=True,
    )

def footer_note():
    st.markdown(
        """
        <div class="footer-note">
        Proof-of-concept note: the current live implementation uses synthetic demo data and a fusion
        demonstration model. Product framing, use cases, pipeline stages, and platform architecture
        are included to match the expected POC presentation experience.
        </div>
        """,
        unsafe_allow_html=True,
    )