import streamlit as st

def section_title(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def hero_section():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-badge">Premium Clinical Demo • Multimodal Dental AI</div>
            <div class="hero-title">🦷 Dental AI Fusion Platform</div>
            <div class="hero-subtitle">
                A product-grade proof-of-concept experience for multimodal dental imaging:
                panoramic X-ray, CBCT, and soft tissue fusion presented through a guided
                clinical dashboard for diagnostics, planning, and product storytelling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_row():
    cols = st.columns(4)
    values = [
        ("Modalities", "3"),
        ("Fusion Layer", "Feature-Level"),
        ("Inference Mode", "Real-Time Demo"),
        ("Deployment Story", "Cloud → Edge"),
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
            <strong>Clinical Workflow:</strong> Input acquisition → preprocessing →
            registration/normalization → 2D/3D feature extraction → fusion →
            reconstruction/visualization → treatment-planning support.
        </div>
        """,
        unsafe_allow_html=True,
    )

def overview_cards():
    section_title("Why This Platform", "Position the product before showing the live demo.")
    cols = st.columns(3)
    items = [
        ("Unified Visualization", "Bring panoramic, CBCT, and soft tissue context into one guided clinical view."),
        ("Decision Support", "Help users move from fragmented imaging review to integrated planning support."),
        ("Deployment Narrative", "Frame the platform as MONAI-based today with future clinic/edge deployment possibilities."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def challenge_cards():
    section_title("Clinical Challenges", "Business and clinical pain points the product addresses.")
    cols = st.columns(3)
    items = [
        ("Fragmented Review", "Clinicians often interpret multiple imaging modalities separately."),
        ("Soft Tissue Gap", "Hard tissue detail is strong, but soft tissue is less unified in planning workflows."),
        ("Time Burden", "Manual correlation across modalities adds time and variability."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def modality_icon_cards():
    section_title("Core Input Modalities", "Styled to feel closer to a product deck than a development page.")
    cols = st.columns(3)
    items = [
        ("🩻", "Panoramic X-ray", "Broad contextual overview of dentition and maxillofacial structures."),
        ("🧊", "CBCT Volume", "Volumetric spatial detail for structural interpretation and planning."),
        ("🫧", "Soft Tissue", "Surface/contour context to complement hard tissue understanding."),
    ]
    for col, (icon, title, text) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="icon-card">
                    <div class="icon-emoji">{icon}</div>
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

def pipeline_cards():
    section_title("Pipeline Architecture", "Show the system as an end-to-end platform, not just a model.")
    pipeline = [
        ("1. Multimodal Acquisition", "Capture panoramic, CBCT, and soft tissue inputs."),
        ("2. Registration & Normalization", "Align and standardize inputs for downstream processing."),
        ("3. Feature Extraction", "Use dedicated 2D and 3D encoders for modality-specific representations."),
        ("4. Fusion Engine", "Combine modality features into a shared latent representation."),
        ("5. Reconstruction / Visualization", "Generate a clinically reviewable fused output."),
        ("6. Clinical Dashboard", "Present results within a decision-support style interface."),
    ]
    for title, text in pipeline:
        st.markdown(
            f'<div class="pipeline-box"><strong>{title}</strong><br>{text}</div>',
            unsafe_allow_html=True,
        )

def upload_console():
    section_title("Clinical Input Console", "Upload preview images to strengthen the product demo feel.")
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
    section_title("Clinical Use Cases", "Match the use-case language expected in the POC.")
    tabs = st.tabs(["Orthodontics", "Implants", "TMJ", "Surgery"])
    with tabs[0]:
        st.markdown("Use fused multimodal context for orthodontic review, communication, and treatment planning support.")
    with tabs[1]:
        st.markdown("Combine bone and surface context to support implant planning workflows.")
    with tabs[2]:
        st.markdown("Position the architecture as extensible for TMJ and dynamic functional analysis.")
    with tabs[3]:
        st.markdown("Use integrated visualization to support pre-operative and surgical simulation narratives.")

def platform_cards():
    section_title("Platform Layers", "Explain the product in architecture language for stakeholders.")
    cols = st.columns(4)
    items = [
        ("MONAI Core", "Medical-imaging-specific transforms, models, and inference support."),
        ("Fusion Engine", "2D panoramic + 3D CBCT + soft tissue fusion layer."),
        ("Clinical UI", "Guided dashboard designed for interpretation and review."),
        ("Deployment", "Cloud POC today with future edge or on-prem positioning."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def architecture_page_cards():
    section_title("Solution Architecture", "A dedicated architecture view adds a strong product-demo feel.")
    row1 = st.columns(3)
    items1 = [
        ("Input Layer", "Panoramic, CBCT, and soft tissue inputs enter through a multimodal acquisition layer."),
        ("Processing Layer", "Normalization, registration framing, and modality-specific preprocessing prepare the inputs."),
        ("AI Layer", "2D and 3D encoders feed a feature-level fusion pipeline that produces a shared output."),
    ]
    for col, (title, text) in zip(row1, items1):
        with col:
            st.markdown(f'<div class="architecture-node"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    row2 = st.columns(3)
    items2 = [
        ("Visualization Layer", "Clinically interpretable fused outputs are presented through a decision-support UI."),
        ("Clinical Layer", "Use-case alignment includes orthodontics, implants, TMJ, and surgery."),
        ("Deployment Layer", "Current cloud POC can be positioned toward future edge, clinic, or device-side deployment."),
    ]
    for col, (title, text) in zip(row2, items2):
        with col:
            st.markdown(f'<div class="architecture-node"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def outcomes_cards():
    section_title("Expected Outcomes", "Use business-facing language alongside the technical demo.")
    cols = st.columns(4)
    items = [
        ("Planning Efficiency", "Reduce manual multi-modality correlation time."),
        ("Better Visualization", "Provide more comprehensive anatomical context."),
        ("Treatment Support", "Support interpretation and planning through integrated output."),
        ("Commercial Readiness", "Present a more product-like POC to clients and stakeholders."),
    ]
    for col, (title, text) in zip(cols, items):
        with col:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def clinical_summary_box():
    st.markdown(
        """
        <div class="clinical-note">
        <strong>Demo Positioning Note</strong><br>
        The live app demonstrates a multimodal fusion experience and clinical dashboard direction.
        Advanced elements such as validated registration, temporal modeling, full DICOM ingestion,
        and production deployment should be described as architecture and roadmap layers unless
        fully implemented and validated.
        </div>
        """,
        unsafe_allow_html=True,
    )

def footer_note():
    st.markdown(
        """
        <div class="footer-note">
        This proof-of-concept uses synthetic demo data and a live fusion visualization workflow
        to represent the broader product architecture described in the project materials.
        </div>
        """,
        unsafe_allow_html=True,
    )