def load_css():
    return """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(56, 189, 248, 0.12), transparent 20%),
            radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.15), transparent 24%),
            linear-gradient(180deg, #050814 0%, #091120 100%);
        color: #eef2ff;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    p, li, div, span {
        color: #d7dceb;
    }

    .hero-shell {
        padding: 2.6rem 2.3rem;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(17,24,39,0.96), rgba(10,15,29,0.97));
        border: 1px solid rgba(129, 140, 248, 0.18);
        box-shadow: 0 24px 70px rgba(0,0,0,0.38);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -40px;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(34,211,238,0.16), transparent 60%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.10);
        border: 1px solid rgba(34, 211, 238, 0.22);
        color: #a5f3fc;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 860;
        line-height: 1.03;
        margin-bottom: 0.7rem;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.75;
        color: #c6d3f5;
        max-width: 940px;
    }

    .section-title {
        font-size: 1.62rem;
        font-weight: 780;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        color: #ffffff;
    }

    .section-subtitle {
        color: #9db1db;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
        font-size: 0.97rem;
    }

    .product-card {
        background: rgba(13, 19, 36, 0.97);
        border: 1px solid rgba(116, 137, 255, 0.14);
        border-radius: 22px;
        padding: 1.15rem 1rem;
        min-height: 180px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .product-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.03rem;
        color: #ffffff;
    }

    .product-card p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #d7dceb;
    }

    .icon-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.97), rgba(10,16,31,0.97));
        border: 1px solid rgba(125, 144, 255, 0.14);
        border-radius: 22px;
        padding: 1rem;
        min-height: 170px;
        margin-bottom: 1rem;
    }

    .icon-emoji {
        font-size: 1.5rem;
        margin-bottom: 0.55rem;
    }

    .mini-tag {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        background: rgba(99,102,241,0.14);
        border: 1px solid rgba(125, 144, 255, 0.18);
        color: #dbeafe;
        font-size: 0.8rem;
    }

    .pipeline-box {
        background: rgba(12,18,34,0.95);
        border-radius: 18px;
        border: 1px solid rgba(110,231,255,0.12);
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #22d3ee;
    }

    .pipeline-box strong {
        color: #ffffff;
    }

    .workflow-banner {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.10), rgba(99,102,241,0.12));
        border: 1px solid rgba(125, 144, 255, 0.16);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .workflow-banner strong {
        color: #ffffff;
    }

    .clinical-note {
        background: rgba(12, 20, 40, 0.95);
        border: 1px solid rgba(125, 144, 255, 0.14);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: #dbe3f7;
        line-height: 1.65;
    }

    .summary-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(11, 18, 34, 0.96));
        border: 1px solid rgba(125, 144, 255, 0.16);
        border-radius: 20px;
        padding: 1rem 1rem;
        margin-bottom: 1rem;
        min-height: 130px;
    }

    .summary-card h4 {
        margin-top: 0;
        margin-bottom: 0.6rem;
        color: #ffffff;
    }

    .upload-box {
        background: rgba(15, 23, 42, 0.92);
        border: 1px dashed rgba(148, 163, 184, 0.32);
        border-radius: 18px;
        padding: 1rem;
    }

    .preview-card {
        background: rgba(13, 19, 36, 0.97);
        border: 1px solid rgba(116, 137, 255, 0.14);
        border-radius: 20px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    .kpi-shell {
        background: rgba(12,18,34,0.96);
        border: 1px solid rgba(125,144,255,0.16);
        border-radius: 20px;
        padding: 0.8rem 1rem;
        text-align: center;
    }

    .kpi-label {
        color: #9db1db;
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 800;
    }

    .architecture-node {
        background: rgba(13,19,36,0.96);
        border: 1px solid rgba(125,144,255,0.16);
        border-radius: 18px;
        padding: 1rem;
        min-height: 150px;
        margin-bottom: 1rem;
    }

    .architecture-node h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }

    div[data-testid="stMetric"] {
        background: rgba(12,18,34,0.95);
        border: 1px solid rgba(125,144,255,0.16);
        border-radius: 18px;
        padding: 0.65rem 0.8rem;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.62rem 1.05rem;
        font-weight: 700;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #0891b2);
        color: white;
    }

    .footer-note {
        margin-top: 2rem;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: rgba(12,20,40,0.92);
        border: 1px solid rgba(125,144,255,0.14);
        font-size: 0.92rem;
        color: #d6d9e0;
    }
    </style>
    """