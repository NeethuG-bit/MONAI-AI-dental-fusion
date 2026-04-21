import streamlit as st

def hero_section():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🦷 Dental AI Fusion Platform</div>
            <div class="hero-subtitle">
                AI-powered multimodal dental imaging demo combining panoramic X-ray,
                CBCT, and soft tissue views into a unified clinical visualization
                experience for diagnostics and treatment planning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_title(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def problem_section():
    section_title("Clinical Challenges")
    cols = st.columns(3)
    cards = [
        ("Fragmented Data Sources", "Clinicians often interpret panoramic, CBCT, and soft-tissue views separately instead of through a unified interface."),
        ("Limited Soft Tissue Context", "Conventional radiographic workflows emphasize hard tissue and may underrepresent soft tissue planning considerations."),
        ("Time-Intensive Analysis", "Manual correlation across modalities increases review time and creates variability in decision-making."),
    ]
    for col, (title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def modalities_section():
    section_title("Input Modalities")
    cols = st.columns(3)
    cards = [
        ("2D Panoramic X-ray", "High-level overview of teeth and maxillofacial structures for broad screening and contextual review."),
        ("3D CBCT Volume", "Volumetric hard-tissue visualization for spatial relationships, bone assessment, and procedural planning."),
        ("Soft Tissue Surface", "Surface-level context supporting esthetic, gingival, and soft-tissue-aware planning."),
    ]
    for col, (title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def pipeline_section():
    section_title("End-to-End Pipeline")
    steps = [
        ("1. Data Acquisition", "Capture panoramic, CBCT, and soft tissue modalities."),
        ("2. Registration & Alignment", "Align imaging modalities into a coherent reference frame."),
        ("3. Intensity Normalization", "Standardize values for robust downstream processing."),
        ("4. Feature Extraction", "Use 2D and 3D neural encoders to derive modality-specific features."),
        ("5. Multimodal Fusion", "Combine features into a shared representation."),
        ("6. 3D Reconstruction / Output", "Generate a fused output for clinical visualization."),
        ("7. Clinical Visualization", "Display interpretable results for treatment planning support."),
    ]
    for title, text in steps:
        st.markdown(
            f'<div class="pipeline-step"><strong>{title}</strong><br>{text}</div>',
            unsafe_allow_html=True
        )

def use_cases_section():
    section_title("Primary Clinical Use Cases")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Orthodontic Planning",
        "Implant Placement",
        "TMJ Analysis",
        "Surgical Simulation",
    ])

    with tab1:
        st.markdown("Integrated hard and soft tissue context for treatment planning, tooth movement review, and outcome communication.")
    with tab2:
        st.markdown("Bone volume and structural context for implant placement planning and procedural safety review.")
    with tab3:
        st.markdown("Potential extension point for multimodal functional and structural TMJ assessment workflows.")
    with tab4:
        st.markdown("Fusion-driven 3D visualization to support pre-operative planning and procedural walkthroughs.")

def platform_section():
    section_title("Platform Architecture")
    cols = st.columns(4)
    cards = [
        ("MONAI Core", "Medical imaging transforms, model building, and inference workflow support."),
        ("Fusion Engine", "Demo model combining 2D panoramic, 3D CBCT, and soft tissue encoders."),
        ("Clinical UI", "Decision-support-style interface designed for guided review rather than raw script output."),
        ("Deployment Readiness", "Cloud demo today; edge or on-prem inference story can be positioned as future deployment path."),
    ]
    for col, (title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def outcomes_section():
    section_title("Expected Outcomes")
    cols = st.columns(4)
    cards = [
        ("Faster Review", "Reduce time spent manually correlating multiple imaging sources."),
        ("Better Visualization", "Present more comprehensive anatomical context in one interface."),
        ("Planning Support", "Support treatment planning workflows with fused representations."),
        ("Product Readiness", "Demonstrate a credible AI-assisted dental imaging POC to clients."),
    ]
    for col, (title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def footer_note():
    st.markdown(
        """
        <div class="footer-note">
        This application is a proof-of-concept demo. The current live implementation focuses on multimodal
        fusion visualization using synthetic demo data, while broader clinical workflow elements are presented
        as product architecture and roadmap components.
        </div>
        """,
        unsafe_allow_html=True,
    )