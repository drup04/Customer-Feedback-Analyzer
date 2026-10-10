import csv
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feedback_logic import (
    FeedbackAnalysisError,
    analyze_feedback,
    example_quote,
    mock_call_llm,
    sentiment_counts,
    top_themes,
)
from llm_client import call_llm

load_dotenv()

# Color Palette
SENTIMENT_COLORS = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}
SEQUENTIAL_BLUE = "#3b82f6"

st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        overflow-y: auto !important;
        height: auto !important;
    }

    [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
        overflow-y: auto !important;
    }

    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 240px !important;
        max-width: 1260px !important;
    }

    footer {
        display: none !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 2px solid rgba(129, 140, 248, 0.45);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 26px;
        box-shadow: 0 12px 32px -4px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 35%, #c084fc 70%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        line-height: 1.15;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 6px 16px;
        border-radius: 9999px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #f1f5f9 !important;
        font-weight: 400;
        max-width: 840px;
        line-height: 1.6;
        margin: 0;
    }

    .highlight-cyan {
        color: #38bdf8 !important;
        font-weight: 700;
    }

    .highlight-emerald {
        color: #34d399 !important;
        font-weight: 700;
    }

    .highlight-amber {
        color: #fbbf24 !important;
        font-weight: 700;
    }

    /* Upload Box Card */
    .upload-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }

    /* Status Badges */
    .badge-demo {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .badge-live {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Quote Cards */
    .quote-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }

    .quote-text {
        font-size: 0.96rem;
        color: #f8fafc !important;
        font-style: italic;
        line-height: 1.5;
        margin-bottom: 6px;
    }

    .quote-author {
        font-size: 0.8rem;
        color: #a5b4fc !important;
        font-weight: 600;
    }

    /* Template Hint */
    .template-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
env_key = os.environ.get("OPENAI_API_KEY", "").strip()
has_env_key = bool(env_key and env_key != "your-key-here")

with st.sidebar:
    st.markdown("### ⚙️ Analyzer Configuration")

    mode = st.radio(
        "AI Engine",
        options=["⚡ Demo Simulation (Offline)", "🤖 Live OpenAI Model"],
        index=1 if has_env_key else 0,
        help="Demo mode processes feedback instantly without API cost. Live mode uses OpenAI.",
    )

    api_key = ""
    selected_model = "gpt-4o-mini"
    if "Live OpenAI" in mode:
        api_key = st.text_input(
            "OpenAI API Key",
            value=env_key if env_key != "your-key-here" else "",
            type="password",
            help="Your key is only kept in memory for this session.",
        )
        selected_model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o-mini is recommended for fast, cost-effective tagging.",
        )
        if not api_key:
            st.warning("⚠️ Enter your OpenAI key or switch to Demo Simulation.")

    batch_size = st.slider(
        "Batch Size (reviews per call)",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
        help="Groups reviews together to preserve theme consistency across prompts.",
    )

    st.markdown("---")
    st.markdown("### 💡 Product Insights")
    st.caption(
        "Converts qualitative feedback into **ranked roadmap themes** "
        "backed by quantifiable customer evidence."
    )

    if st.button("🧹 Clear Analysis Data", use_container_width=True):
        if "analyzed" in st.session_state:
            del st.session_state["analyzed"]
            st.rerun()

# Hero Section
col_hero, col_status = st.columns([4, 1])
with col_hero:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Customer Feedback Analyzer</div>
            <div class="hero-badge">✨ AI-Powered Product Intelligence</div>
            <p class="hero-subtitle">
                Upload raw customer reviews, support tickets, or survey comments. 
                Extract actionable <span class="highlight-cyan">roadmap themes</span>, <span class="highlight-emerald">sentiment breakdowns</span>, and <span class="highlight-amber">direct quotes</span> automatically.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_status:
    st.write("")
    st.write("")
    if "Demo" in mode:
        st.markdown('<div class="badge-demo">⚡ Demo Engine</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge-live">🟢 Live API ({selected_model})</div>', unsafe_allow_html=True)

# Upload & Input Section (No sequential numbering)
st.markdown("### Upload Customer Feedback")

upload_container = st.container()
with upload_container:
    col_upload, col_actions = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Select a CSV file containing user feedback",
            type=["csv"],
            help="Requires a text column named 'review_text', 'feedback', 'comment', or 'text'.",
        )

    with col_actions:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 Analyze Feedback", type="primary", use_container_width=True)

        # Starter template download for convenience
        sample_template = (
            "review_id,rating,review_text\n"
            "1,2,\"The dashboard crashes whenever exporting files over 500 rows.\"\n"
            "2,5,\"Support team fixed my login issue in less than ten minutes.\"\n"
            "3,1,\"Billed for seats we never used, pricing is not clear on the website.\"\n"
            "4,4,\"Onboarding checklist made setup easy and intuitive.\"\n"
            "5,2,\"Load times are slow in the morning during peak reporting hours.\"\n"
        )
        st.download_button(
            "📥 CSV Template",
            data=sample_template,
            file_name="feedback_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a clean 5-row CSV template to test or format your data.",
        )

# Trigger Analysis
if analyze_btn:
    if uploaded is not None:
        csv_text = uploaded.getvalue().decode("utf-8")
    else:
        st.error("Please upload a CSV file with customer feedback.")
        st.stop()

    # Determine caller
    is_demo = "Demo" in mode
    if not is_demo:
        effective_key = api_key or env_key
        if not effective_key:
            st.error("OpenAI API key missing. Enter your key in the sidebar or switch to Demo Simulation.")
            st.stop()

        def custom_call_llm(prompt: str) -> str:
            return call_llm(prompt, model=selected_model, api_key=effective_key)

        llm_fn = custom_call_llm
    else:
        llm_fn = mock_call_llm

    progress_bar = st.progress(0, text="Analyzing customer feedback...")

    def update_progress(current_batch: int, total_batches: int):
        pct = int((current_batch / total_batches) * 100) if total_batches > 0 else 100
        progress_bar.progress(
            min(pct, 100),
            text=f"Processing batch {min(current_batch + 1, total_batches)} of {total_batches}...",
        )

    try:
        results = analyze_feedback(
            csv_text,
            call_llm=llm_fn,
            batch_size=batch_size,
            progress_callback=update_progress,
        )
        st.session_state.analyzed = results
        progress_bar.empty()
        st.success(f"Analysis complete! Tagged {len(results)} customer feedback items.")
    except FeedbackAnalysisError as e:
        progress_bar.empty()
        st.error(f"Analysis Error: {e}")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Unexpected Error: {e}")

# Render Results (No sequential numbering)
analyzed = st.session_state.get("analyzed")

if analyzed:
    st.divider()
    st.markdown("### Executive Summary & Key Metrics")

    counts = sentiment_counts(analyzed)
    total = sum(counts.values())
    themes_ranked = top_themes(analyzed, n=15)
    top_theme_name = themes_ranked[0][0].title() if themes_ranked else "N/A"

    # Executive KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Analyzed", f"{total}")
    pos_pct = (counts.get("positive", 0) / total * 100) if total else 0
    neu_pct = (counts.get("neutral", 0) / total * 100) if total else 0
    neg_pct = (counts.get("negative", 0) / total * 100) if total else 0
    kpi2.metric("Positive", f"{counts.get('positive', 0)}", f"{pos_pct:.0f}%")
    kpi3.metric("Neutral", f"{counts.get('neutral', 0)}", f"{neu_pct:.0f}%")
    kpi4.metric("Negative", f"{counts.get('negative', 0)}", f"{neg_pct:.0f}%")
    kpi5.metric("Top Theme", f"{top_theme_name}", f"{len(themes_ranked)} unique")

    st.write("")

    # Clean Tabbed Results
    tab_charts, tab_quotes, tab_explorer, tab_export = st.tabs([
        "📊 Visual Breakdown",
        "📌 Evidence & Quotes",
        "🔍 Filter & Explore Reviews",
        "💾 Export Dataset",
    ])

    with tab_charts:
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("#### Sentiment Breakdown")
            fig, ax = plt.subplots(figsize=(6, 2.6))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            labels = [s.title() for s in counts]
            values = list(counts.values())
            colors = [SENTIMENT_COLORS[s] for s in counts]
            bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")
            ax.bar_label(
                bars,
                labels=[f"{v} ({v/total*100:.0f}%)" if total else f"{v}" for v in values],
                padding=6,
                color="#ffffff",
                fontweight="bold",
                fontsize=9,
            )
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(axis="both", length=0, labelsize=10, colors="#f1f5f9")
            ax.set_xticks([])
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)

        with chart_col2:
            st.markdown("#### Top Product Themes")
            if themes_ranked:
                top_display = themes_ranked[:8]
                fig2, ax2 = plt.subplots(figsize=(6, max(2.6, 0.35 * len(top_display))))
                fig2.patch.set_facecolor('none')
                ax2.set_facecolor('none')
                names = [t.title() for t, _ in reversed(top_display)]
                counts_list = [c for _, c in reversed(top_display)]
                bars2 = ax2.barh(names, counts_list, color=SEQUENTIAL_BLUE, height=0.6, edgecolor="none")
                ax2.bar_label(bars2, padding=6, color="#ffffff", fontweight="bold", fontsize=9)
                ax2.spines[["top", "right", "bottom", "left"]].set_visible(False)
                ax2.tick_params(axis="both", length=0, labelsize=10, colors="#f1f5f9")
                ax2.set_xticks([])
                fig2.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No themes extracted.")

    with tab_quotes:
        st.caption("Verbatim customer quotes matched to each top theme for roadmap reviews:")
        quote_cols = st.columns(2)
        for idx, (theme, count) in enumerate(themes_ranked[:8]):
            col_idx = idx % 2
            with quote_cols[col_idx]:
                with st.expander(f"📌 {theme.title()} — {count} mention(s)", expanded=(idx < 2)):
                    quote = example_quote(analyzed, theme)
                    st.markdown(
                        f"""
                        <div class="quote-card">
                            <div class="quote-text">"{quote}"</div>
                            <div class="quote-author">Tagged Theme: {theme.title()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with tab_explorer:
        fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
        with fcol1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=["All", "Positive", "Neutral", "Negative"],
                key="explorer_sentiment",
            )
        with fcol2:
            theme_options = ["All"] + [t.title() for t, _ in themes_ranked]
            theme_filter = st.selectbox(
                "Filter by Theme",
                options=theme_options,
                key="explorer_theme",
            )
        with fcol3:
            search_query = st.text_input(
                "Search in text",
                placeholder="Type keywords...",
                key="explorer_search",
            )

        # Filter rows
        filtered = []
        for item in analyzed:
            if sentiment_filter != "All" and item.sentiment.lower() != sentiment_filter.lower():
                continue
            if theme_filter != "All" and theme_filter.lower() not in [t.lower() for t in item.themes]:
                continue
            if search_query and search_query.lower() not in item.text.lower():
                continue
            filtered.append(item)

        st.caption(f"Showing **{len(filtered)}** of {len(analyzed)} reviews")

        # Feedback text first, ID at the end
        table_data = [
            {
                "Feedback Review": str(item.text),
                "Sentiment": str(item.sentiment).upper(),
                "Themes": ", ".join(item.themes),
                "Rating": str(item.rating) if item.rating else "-",
                "Ref ID": str(item.id),
            }
            for item in filtered
        ]
        if table_data:
            df_display = pd.DataFrame(table_data)
            df_display.columns = [str(c) for c in df_display.columns]
            st.dataframe(df_display, height=360, hide_index=True)
        else:
            st.info("No feedback rows match the active filters.")

    with tab_export:
        st.markdown("#### Export Enriched Dataset")
        st.write("Download the complete customer feedback dataset with model-generated sentiment and theme tags.")
        export_rows = [
            {
                "review_text": item.text,
                "sentiment": item.sentiment,
                "themes": "; ".join(item.themes),
                "rating": item.rating,
                "review_id": item.id,
            }
            for item in analyzed
        ]
        export_df = pd.DataFrame(export_rows)
        export_csv = export_df.to_csv(index=False).encode("utf-8")

        st.dataframe(export_df.head(10), height=240, hide_index=True)

        st.download_button(
            "💾 Download Analyzed CSV (analyzed_customer_feedback.csv)",
            data=export_csv,
            file_name="analyzed_customer_feedback.csv",
            mime="text/csv",
            type="primary",
        )

    # Bottom spacer
    st.markdown("<div style='height: 120px; padding-bottom: 60px;'></div>", unsafe_allow_html=True)
