import streamlit as st

def section_title(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def hero_section():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-badge">AUMTECH-style POC • Clinical Product Experience</div>
            <div class="hero-title">🦷 Dental AI Fusion Platform</div>
            <div class="hero-subtitle">
                Product-oriented multimodal fusion experience combining panoramic X-ray,
                CBCT, and soft tissue views into a single clinical dashboard for
                diagnostics, treatment planning, and product demonstration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_row():
    cols = st.columns(4)
    values = [
        ("Modalities", "3 Inputs"),
        ("Fusion Strategy", "Feature-Level"),
        ("Inference", "Real-Time Demo"),
        ("Mode", "Clinical POC"),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="kpi-shell">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def workflow_banner():
    st.markdown(
        """
        <div class="workflow-banner">
            <strong>Workflow:</strong> Acquire multimodal inputs → normalize and align →
            extract 2D/3D features → fuse into shared representation → visualize
            clinically meaningful output.
        </div>
        """,
        unsafe_allow_html=True,
    )

def overview_cards():
    section_title(
        "Executive Overview",
        "Use this page to explain the product story before showing the live model."
    )
    cols = st.columns(3)
    items = [
        ("Unified Visualization", "Present hard tissue and soft tissue context within one guided review experience."),
        ("Clinical Workflow Support", "Reduce manual cross-referencing across modalities and improve planning efficiency."),
        ("Deployment Narrative", "Position the solution as MONAI-based with future edge or clinical deployment pathways."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def challenge_cards():
    section_title(
        "Clinical Challenges",
        "This mirrors the business and clinical framing in the PDF."
    )
    cols = st.columns(3)
    items = [
        ("Fragmented Data Sources", "Panoramic, CBCT, and soft tissue modalities are often reviewed independently."),
        ("Limited Soft Tissue Context", "Treatment planning benefits from richer integration of hard and soft tissue information."),
        ("Time-Intensive Analysis", "Manual synthesis of multiple image sources adds interpretation time and variability."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def modalities_cards():
    section_title(
        "Imaging Modalities",
        "The current demo uses synthetic data to simulate these sources."
    )
    cols = st.columns(3)
    items = [
        ("2D Panoramic X-ray", "High-level contextual overview of dental and maxillofacial structures."),
        ("3D CBCT Volume", "Volumetric structural view for spatial interpretation and planning."),
        ("Soft Tissue Surface", "Surface-aware context supporting esthetic and soft tissue considerations."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def pipeline_cards():
    section_title(
        "Pipeline Architecture",
        "Present the product as an end-to-end workflow, not just a model runner."
    )
    pipeline = [
        ("1. Acquisition", "Collect panoramic, CBCT, and soft tissue inputs."),
        ("2. Registration", "Align modalities into a common workflow representation."),
        ("3. Normalization", "Standardize inputs for consistent model processing."),
        ("4. Feature Extraction", "Use dedicated encoders for modality-specific features."),
        ("5. Fusion", "Combine features into a shared multimodal representation."),
        ("6. Reconstruction / Output", "Generate an interpretable fused output."),
        ("7. Clinical Review", "Display results in a guided decision-support interface."),
    ]
    for title, text in pipeline:
        st.markdown(
            f'<div class="pipeline-box"><strong>{title}</strong><br>{text}</div>',
            unsafe_allow_html=True,
        )

def upload_placeholders():
    section_title(
        "Clinical Input Console",
        "Use uploaded previews if available; otherwise the app falls back to synthetic demo data."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="upload-box">Panoramic Input</div>', unsafe_allow_html=True)
        pan_file = st.file_uploader("Upload panoramic image", type=["png", "jpg", "jpeg"], key="pan_upload")
    with c2:
        st.markdown('<div class="upload-box">CBCT Preview</div>', unsafe_allow_html=True)
        cbct_file = st.file_uploader("Upload CBCT preview image", type=["png", "jpg", "jpeg"], key="cbct_upload")
    with c3:
        st.markdown('<div class="upload-box">Soft Tissue Input</div>', unsafe_allow_html=True)
        soft_file = st.file_uploader("Upload soft tissue image", type=["png", "jpg", "jpeg"], key="soft_upload")

    return pan_file, cbct_file, soft_file

def use_case_tabs():
    section_title(
        "Clinical Use Cases",
        "Map the platform to the exact use-case language from the POC."
    )
    tabs = st.tabs([
        "Orthodontic Planning",
        "Implant Placement",
        "TMJ Analysis",
        "Surgical Simulation",
    ])

    with tabs[0]:
        st.markdown("Integrated hard and soft tissue context for planning, review, and treatment communication.")
    with tabs[1]:
        st.markdown("3D bone context plus fused visualization to support implant planning decisions.")
    with tabs[2]:
        st.markdown("Position the system as extensible for multimodal and functional analysis workflows.")
    with tabs[3]:
        st.markdown("Use fused visualizations to support procedural planning and clinical walkthroughs.")

def platform_cards():
    section_title(
        "Platform Layers",
        "Explain the system in product architecture language during the client demo."
    )
    cols = st.columns(4)
    items = [
        ("MONAI Layer", "Medical imaging transforms, architecture support, and inference workflow."),
        ("Fusion Layer", "2D + 3D + soft tissue feature fusion into a shared output representation."),
        ("Clinical UI Layer", "Decision-support interface for clinicians, operators, and demo stakeholders."),
        ("Deployment Layer", "Cloud-hosted POC today with edge/on-prem positioning as future pathway."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def outcomes_cards():
    section_title(
        "Expected Outcomes",
        "Use these cards to reinforce client-facing value."
    )
    cols = st.columns(4)
    items = [
        ("Faster Planning", "Reduce manual cross-modality review time."),
        ("Improved Visualization", "Provide more comprehensive multimodal anatomical context."),
        ("Clinical Support", "Support treatment planning with guided review outputs."),
        ("POC Credibility", "Show a product direction that feels closer to deployment."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def clinical_summary_box():
    st.markdown(
        """
        <div class="clinical-note">
        <strong>Clinical Product Positioning</strong><br>
        This app demonstrates the fusion workflow, clinical UI direction, and decision-support experience.
        Advanced elements such as validated registration, temporal modeling, device integration, and
        production deployment can be presented as architecture and roadmap layers rather than overstated
        as fully live features.
        </div>
        """,
        unsafe_allow_html=True,
    )

def footer_note():
    st.markdown(
        """
        <div class="footer-note">
        Proof-of-concept note: this live implementation uses synthetic demo data for fusion visualization.
        Product framing, workflow sections, and clinical use-case positioning are included to match the
        expected demo and presentation experience.
        </div>
        """,
        unsafe_allow_html=True,
    )