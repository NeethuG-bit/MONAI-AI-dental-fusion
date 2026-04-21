def load_css():
    return """
    <style>
    .stApp {
        background: linear-gradient(180deg, #060b16 0%, #0b1020 100%);
        color: #f3f5f7;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    p, li, label, .stMarkdown, .stText {
        color: #d6d9e0 !important;
    }

    .hero-box {
        padding: 2.2rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(20,28,55,0.95), rgba(12,18,34,0.95));
        border: 1px solid rgba(120, 140, 255, 0.22);
        box-shadow: 0 12px 40px rgba(0,0,0,0.35);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #c7d2fe;
        line-height: 1.65;
        max-width: 900px;
    }

    .section-title {
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }

    .card {
        background: rgba(16, 24, 45, 0.92);
        border: 1px solid rgba(120, 140, 255, 0.18);
        border-radius: 20px;
        padding: 1.1rem 1rem;
        min-height: 180px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
    }

    .card h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #ffffff;
        font-size: 1.05rem;
    }

    .card p {
        margin: 0;
        color: #d6d9e0;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .metric-card {
        background: rgba(12, 20, 40, 0.95);
        border: 1px solid rgba(110, 231, 255, 0.16);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        text-align: center;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9fb0d8;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
    }

    .pipeline-step {
        background: rgba(20, 28, 50, 0.95);
        border-left: 4px solid #6ee7ff;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
    }

    .pipeline-step strong {
        color: #ffffff;
    }

    .tag {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border-radius: 999px;
        background: rgba(86, 104, 255, 0.16);
        border: 1px solid rgba(120, 140, 255, 0.25);
        color: #dbeafe;
        font-size: 0.82rem;
    }

    .footer-note {
        margin-top: 2rem;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: rgba(12, 20, 40, 0.9);
        border: 1px solid rgba(120, 140, 255, 0.14);
        color: #cbd5e1;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(16, 24, 45, 0.95);
        border: 1px solid rgba(120, 140, 255, 0.18);
        padding: 0.6rem 0.8rem;
        border-radius: 16px;
    }

    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 700;
    }

    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #0891b2);
        color: white;
    }

    hr {
        border-color: rgba(120, 140, 255, 0.18);
    }
    </style>
    """