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
    section_title("Overview", "Product summary")
    c1, c2, c3 = st.columns(3)
    c1.info("Unified visualization of panoramic, CBCT, and soft tissue inputs.")
    c2.info("Decision-support framing for treatment planning workflows.")
    c3.info("MONAI-based proof-of-concept with product-style UI.")

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
    section_title("Clinical Use Cases")
    t1, t2, t3, t4 = st.tabs(["Orthodontics", "Implants", "TMJ", "Surgery"])
    with t1:
        st.write("Orthodontic planning support")
    with t2:
        st.write("Implant planning support")
    with t3:
        st.write("TMJ workflow positioning")
    with t4:
        st.write("Surgical simulation positioning")

def platform_cards():
    section_title("Platform")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("MONAI Core")
    c2.info("Fusion Engine")
    c3.info("Clinical UI")
    c4.info("Deployment Layer")

def architecture_page_cards():
    section_title("Architecture")
    c1, c2, c3 = st.columns(3)
    c1.info("Input Layer")
    c2.info("AI Processing Layer")
    c3.info("Visualization Layer")

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