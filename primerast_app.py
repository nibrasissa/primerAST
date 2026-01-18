import io
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from primer_pipeline import (
    run_primer_design_pipeline_for_variant,
    run_primer_design_pipeline_from_sequence,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "primersat_gb_model.joblib"
META_PATH = ROOT / "primersat_metadata.json"


@st.cache_resource
def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with META_PATH.open() as f:
        meta = json.load(f)
    return model, meta


def score_primers(
    df_primers: pd.DataFrame,
    model,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Score primer pairs using the trained ML model."""
    missing = set(feature_columns) - set(df_primers.columns)
    if missing:
        raise ValueError(f"Missing feature columns in primer DataFrame: {missing}")

    X = df_primers[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    probs = model.predict_proba(X)[:, 1]
    labels = model.predict(X)

    df = df_primers.copy()
    df["PrimerAST_score"] = probs
    df["Predicted_label"] = labels
    df["Predicted_class"] = df["Predicted_label"].map(
        {0: "Predicted to fail", 1: "Validated"}
    )
    return df.sort_values("PrimerAST_score", ascending=False).reset_index(drop=True)


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(
    page_title="PrimerAST:Primer Assessment & Scoring Tool",
    page_icon=None,
    layout="wide",
)

# Top banner
st.markdown(
    """
    <div style="padding: 1.2rem 1.5rem; border-radius: 0.8rem;
                background: linear-gradient(90deg, #1a73e8, #4f8df5);
                color: white; margin-bottom: 1.0rem;">
        <h2 style="margin: 0; font-weight: 600;">
            Primer Scoring and Assessment Tool (PrimerAST)
        </h2>
        <p style="margin: 0.3rem 0 0; font-size: 0.95rem;">
            Variant-aware primer design and machine-learning-based evaluation for diagnostic PCR.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("PrimerAST")
st.sidebar.caption("Primer Scoring and Assessment Tool")

st.sidebar.markdown("### Pipeline overview")
st.sidebar.markdown(
    """
- Variant-aware sequence retrieval (Ensembl)
- Primer3 design with SEQUENCE_TARGET
- QC and thermodynamic feature extraction
- Machine-learning-based scoring (Validated vs Predicted to fail)
"""
)

# Load model and metadata
try:
    model, meta = load_model_and_meta()
    feature_columns = meta["feature_columns"]
except Exception as e:
    st.error(
        "Could not load the trained model or metadata.\n\n"
        f"Details: {e}"
    )
    st.stop()

# Tabs
tab_single, tab_upload, tab_results = st.tabs(
    ["Single Variant", "Batch Variant Report", "Scored Primers"]
)

# ==============================
# 1. Single variant
# ==============================
with tab_single:
    st.subheader("Single Variant Design and Scoring")

    col1, col2, col3 = st.columns(3)
    with col1:
        chrom = st.text_input("Chromosome", value="1", help="e.g. 1, 2, X, Y")
        pos = st.text_input("Position (1-based)", value="")
    with col2:
        ref = st.text_input("Reference allele", value="")
        alt = st.text_input("Alternate allele", value="")
    with col3:
        assembly = st.selectbox("Assembly", ["GRCh38", "GRCh37"], index=0)
        refseq = st.text_input(
            "RefSeq transcript (optional)",
            value="",
            help="e.g. NM_000123.4",
        )

    flank_size = st.slider(
        "Flanking size around the variant (bp upstream and downstream)",
        min_value=200,
        max_value=1500,
        value=500,
        step=50,
    )

    run_single = st.button("Run PrimerAST for this variant", type="primary")

    if run_single:
        if not pos or not ref or not alt:
            st.error("Please fill in at least position, reference allele, and alternate allele.")
        else:
            with st.spinner("Fetching region, designing targeted primers and scoring…"):
                try:
                    df_primers = run_primer_design_pipeline_for_variant(
                        chromosome=chrom,
                        position=int(pos),
                        flank=int(flank_size),
                        assembly=assembly,
                    )
                    if df_primers is None or df_primers.empty:
                        st.warning(
                            "No primer pairs could be designed that cover this variant with "
                            "the current constraints."
                        )
                    else:
                        df_scored = score_primers(df_primers, model, feature_columns)
                        variant_id = f"{chrom}:{pos}_{ref}>{alt}"
                        df_scored["variant_id"] = variant_id
                        df_scored["assembly"] = assembly
                        if refseq:
                            df_scored["refseq"] = refseq

                        st.session_state["scored_single"] = df_scored

                        st.success(
                            "PrimerAST has completed scoring for this variant "
                            "(amplicons constrained to cover the variant locus)."
                        )

                        key_cols = [
                            "variant_id",
                            "primer_id",
                            "Predicted_class",
                            "PrimerAST_score",
                            "amplicon_size",
                            "forward_seq",
                            "reverse_seq",
                        ]
                        cols_present = [c for c in key_cols if c in df_scored.columns]
                        other_cols = [c for c in df_scored.columns if c not in cols_present]

                        st.dataframe(
                            df_scored[cols_present + other_cols],
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Failed to design or score primers: {e}")

# ==============================
# 2. Batch variant report
# ==============================
with tab_upload:
    st.subheader("Batch Variant Report")

    st.markdown(
        """
Upload a CSV or Excel file with basic variant information.

**Required columns (if no `flank_sequence` is present):**

- `variant_id` – any identifier (e.g. chr:pos_ref>alt, rsID, lab code)  
- `chromosome` – e.g. 1, 2, X  
- `position` – genomic coordinate (1-based)  
- `ref` – reference allele  
- `alt` – alternate allele  
- `assembly` – e.g. GRCh38 or GRCh37  

If your file already contains a `flank_sequence` column, the app can use
`run_primer_design_pipeline_from_sequence` as a fallback.
"""
    )

    file = st.file_uploader(
        "Upload variant report (CSV or Excel)",
        type=["csv", "xlsx"],
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        run_batch = st.button("Run PrimerAST on all variants", type="primary")
    with col_right:
        st.write("")

    df_variants = None
    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df_variants = pd.read_csv(file)
            else:
                df_variants = pd.read_excel(file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_variants = None

    if df_variants is not None:
        st.markdown("### Preview of variant report")
        st.dataframe(df_variants.head(), use_container_width=True)

        if run_batch:
            all_scored = []

            if "variant_id" not in df_variants.columns:
                df_variants["variant_id"] = [
                    f"{row.get('chromosome', 'chr?')}:{row.get('position', '?')}_"
                    f"{row.get('ref', '?')}>{row.get('alt', '?')}"
                    for _, row in df_variants.iterrows()
                ]

            with st.spinner("Designing targeted primers and scoring for each variant…"):
                for idx, row in df_variants.iterrows():
                    try:
                        if (
                            "flank_sequence" in df_variants.columns
                            and isinstance(row.get("flank_sequence", ""), str)
                            and len(str(row["flank_sequence"]).strip()) > 80
                        ):
                            df_primers = run_primer_design_pipeline_from_sequence(
                                str(row["flank_sequence"])
                            )
                        else:
                            df_primers = run_primer_design_pipeline_for_variant(
                                chromosome=row["chromosome"],
                                position=int(row["position"]),
                                flank=500,
                                assembly=row.get("assembly", "GRCh38"),
                            )

                        if df_primers is None or df_primers.empty:
                            continue

                        df_scored = score_primers(df_primers, model, feature_columns)
                        df_scored["variant_id"] = row["variant_id"]
                        if "assembly" in df_variants.columns:
                            df_scored["assembly"] = row["assembly"]
                        if "refseq" in df_variants.columns:
                            df_scored["refseq"] = row["refseq"]
                        all_scored.append(df_scored)
                    except Exception as e:
                        st.warning(
                            f"Variant {row.get('variant_id', idx)} failed: {e}"
                        )
                        continue

            if not all_scored:
                st.warning(
                    "No primer sets could be generated or scored from the uploaded variants."
                )
            else:
                scored_all = pd.concat(all_scored, ignore_index=True)
                st.session_state["scored_all"] = scored_all
                st.success(
                    f"PrimerAST completed scoring for {len(all_scored)} variants "
                    "with at least one primer set."
                )
                st.info(
                    "Switch to the 'Scored Primers' tab to explore the results."
                )

# ==============================
# 3. Results / scored primers
# ==============================
with tab_results:
    st.subheader("Scored Primer Sets")

    if "scored_all" not in st.session_state and "scored_single" not in st.session_state:
        st.info(
            "No scored primers are available yet. Use either 'Single Variant' or "
            "'Batch Variant Report' to generate primer sets."
        )
    else:
        frames = []
        if "scored_all" in st.session_state:
            frames.append(st.session_state["scored_all"])
        if "scored_single" in st.session_state:
            frames.append(st.session_state["scored_single"])
        scored_all = pd.concat(frames, ignore_index=True)

        key_cols = [
            "variant_id",
            "primer_id",
            "Predicted_class",
            "PrimerAST_score",
            "amplicon_size",
            "forward_seq",
            "reverse_seq",
        ]
        cols_present = [c for c in key_cols if c in scored_all.columns]
        other_cols = [c for c in scored_all.columns if c not in cols_present]

        variants = sorted(scored_all["variant_id"].unique().tolist())
        selected_variant = st.selectbox(
            "Filter by variant (variant_id)",
            options=["All variants"] + variants,
        )

        if selected_variant != "All variants":
            df_view = scored_all[scored_all["variant_id"] == selected_variant]
        else:
            df_view = scored_all

        st.markdown("### Ranked primers")
        st.dataframe(
            df_view[cols_present + other_cols],
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total primer pairs", df_view.shape[0])
        with col2:
            n_val = int((df_view["Predicted_class"] == "Validated").sum())
            st.metric("Predicted Validated", n_val)
        with col3:
            if df_view.shape[0] > 0:
                mean_score = df_view["PrimerAST_score"].mean()
                st.metric("Mean PrimerAST score", f"{mean_score:.3f}")
            else:
                st.metric("Mean PrimerAST score", "–")

        csv_buf = io.StringIO()
        scored_all.to_csv(csv_buf, index=False)
        st.download_button(
            "Download all scored primers (CSV)",
            data=csv_buf.getvalue(),
            file_name="primerast_scored_primers_by_variant.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Primer Scoring and Assessment Tool (PrimerAST) · "
    "Variant-aware primer design and machine-learning-based evaluation."
)
