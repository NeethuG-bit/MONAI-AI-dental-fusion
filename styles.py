def load_css():
    return """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(34,211,238,0.10), transparent 22%),
            radial-gradient(circle at top left, rgba(99,102,241,0.12), transparent 24%),
            linear-gradient(180deg, #060914 0%, #0a1020 100%);
        color: #edf2f7;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.6rem;
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
        padding: 2.2rem 2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(16,24,45,0.96), rgba(9,14,30,0.96));
        border: 1px solid rgba(125, 144, 255, 0.18);
        box-shadow: 0 16px 50px rgba(0,0,0,0.30);
        margin-bottom: 1.5rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.12);
        border: 1px solid rgba(34, 211, 238, 0.20);
        color: #a5f3fc;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 850;
        line-height: 1.05;
        margin-bottom: 0.7rem;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.06rem;
        line-height: 1.7;
        color: #c6d3f5;
        max-width: 900px;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
        color: #ffffff;
    }

    .section-subtitle {
        color: #9db1db;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
        font-size: 0.96rem;
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

    div[data-testid="stMetric"] {
        background: rgba(12,18,34,0.95);
        border: 1px solid rgba(125,144,255,0.16);
        border-radius: 18px;
        padding: 0.65rem 0.8rem;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.62rem 1.05rem;
        font-weight: 700;
    }

    div.stButton > button:hover {
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