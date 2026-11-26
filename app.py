"""Streamlit Interactive Interoperability Clustering Dashboard.

SIH26129: PCA-Based Dimensionality Reduction of Fragmented Government Service-Usage Data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.decomposition import PCA
import streamlit as st

from src.cluster import compute_intrinsic_metrics
from src.data_gen import CATEGORIES, load_config
from src.overlap import compute_pairwise_overlap_matrix, generate_recommendations

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GovPlatform Interoperability Suite · SIH26129",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM ENTERPRISE GOVTECH CSS STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Clean Font Stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.4);
    }
    .hero-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.5);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 0px;
    }

    /* KPI Summary Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    @media (prefers-color-scheme: dark) {
        .metric-value {
            color: #f8fafc;
        }
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 500;
    }

    /* Action Badges */
    .badge-merge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-gateway {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-sso {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-coexist {
        background-color: #d1fae5;
        color: #047857;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Subheader Stylings */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-top: 10px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_artifacts():
    cfg = load_config()
    platforms_df = pd.read_csv(cfg["paths"]["platforms_csv"])
    feat_df = pd.read_csv(cfg["paths"]["features_csv"])
    scree_df = pd.read_csv(cfg["paths"]["scree_plot_csv"])
    loadings_df = pd.read_csv(cfg["paths"]["pca_loadings_csv"])
    benchmark_df = pd.read_csv(cfg["paths"]["benchmark_csv"])
    recs_df = pd.read_csv(cfg["paths"]["recommendations_csv"])
    cluster_df = pd.read_csv(cfg["paths"]["cluster_results_csv"])
    overlap_df = pd.read_csv(cfg["paths"]["overlap_matrix_csv"], index_col=0)

    feature_cols = [c for c in feat_df.columns if c not in ["platform_id", "platform_name"]]
    X_raw = feat_df[feature_cols].to_numpy(dtype=float)

    return {
        "config": cfg,
        "platforms_df": platforms_df,
        "feat_df": feat_df,
        "feature_cols": feature_cols,
        "X_raw": X_raw,
        "scree_df": scree_df,
        "loadings_df": loadings_df,
        "benchmark_df": benchmark_df,
        "recs_df": recs_df,
        "cluster_df": cluster_df,
        "overlap_df": overlap_df,
    }


def main():
    data = load_all_artifacts()
    cfg = data["config"]
    platforms_df = data["platforms_df"]
    X_raw = data["X_raw"]
    feature_cols = data["feature_cols"]

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ **Clustering & PCA Engine**")
        st.caption("Adjust dimensionality and clustering algorithms in real time.")

        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "Agglomerative (Ward)", "HDBSCAN (Density)"],
            index=0,
            help="Select the partitioning or density algorithm used on the reduced PCA features.",
        )

        pca_variance = st.slider(
            "PCA Variance Threshold",
            min_value=0.70,
            max_value=0.99,
            value=0.92,
            step=0.01,
            help="Determines the number of principal components dynamically chosen based on cumulative explained variance.",
        )

        k_clusters = st.slider(
            "Cluster Count (k)",
            min_value=3,
            max_value=12,
            value=7,
            disabled=(algorithm == "HDBSCAN (Density)"),
            help="Applicable to K-Means and Agglomerative clustering.",
        )

        st.markdown("---")
        st.markdown("### 🗺️ **Visualization Settings**")
        proj_dimension = st.radio(
            "Projection Space",
            ["2D Projection", "3D Interactive Space"],
            index=0,
            horizontal=True,
        )

        proj_type = st.selectbox(
            "Dimensionality Method",
            ["PCA (Principal Components)", "UMAP (Non-linear Manifold)"],
            index=0,
            help="PCA preserves global variance; UMAP preserves local topological neighborhoods.",
        )

        st.markdown("---")
        st.caption(
            "**Smart India Hackathon (SIH26129)**\n\n"
            "• High-Dimensional Service Ingestion\n"
            "• Variance-Preserving PCA Reduction\n"
            "• Interoperability Redundancy Clustering"
        )

    # -------------------------------------------------------------------------
    # DYNAMIC PCA & CLUSTERING COMPUTATION
    # -------------------------------------------------------------------------
    pca_full = PCA(random_state=42).fit(X_raw)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    dyn_n_comps = max(3, int(np.where(cum_var >= pca_variance)[0][0] + 1))
    X_pca_dyn = PCA(n_components=dyn_n_comps, random_state=42).fit_transform(X_raw)

    if algorithm == "K-Means":
        active_labels = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit_predict(X_pca_dyn)
    elif algorithm == "Agglomerative (Ward)":
        active_labels = AgglomerativeClustering(n_clusters=k_clusters, linkage="ward").fit_predict(X_pca_dyn)
    else:
        active_labels = HDBSCAN(min_cluster_size=3, min_samples=2, copy=True).fit_predict(X_pca_dyn)

    metrics = compute_intrinsic_metrics(X_pca_dyn, active_labels)

    # -------------------------------------------------------------------------
    # HERO SECTION
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-tag">Smart India Hackathon · SIH26129</span>
            <div class="hero-title">🏛️ Interoperability Clustering of Government Digital Platforms</div>
            <div class="hero-subtitle">
                Automated high-dimensional feature synthesis, variance-preserving PCA dimensionality reduction,
                and unsupervised clustering to detect cross-departmental service overlaps and consolidation targets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Warning / Status Banner
    is_synthetic = bool(platforms_df["is_synthetic"].iloc[0]) if "is_synthetic" in platforms_df.columns else True
    if is_synthetic:
        st.info(
            "ℹ️ **Evaluation Dataset**: Active simulation across **49 platforms**, **210 digital services**, and **15 administrative categories**. "
            "Planted ground truth clusters are strictly segregated for ARI/NMI external validation.",
            icon="📊",
        )

    # -------------------------------------------------------------------------
    # TOP KPI METRICS BAR
    # -------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">PCA Dimensions</div>
                <div class="metric-value">{dyn_n_comps} <span style="font-size:0.9rem;color:#64748b;">/ {X_raw.shape[1]}</span></div>
                <div class="metric-sub">↓ {(1 - dyn_n_comps/X_raw.shape[1])*100:.1f}% Compression</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Preserved Variance</div>
                <div class="metric-value">{cum_var[dyn_n_comps-1]*100:.1f}%</div>
                <div class="metric-sub">Target: ≥ {pca_variance*100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Active Clusters</div>
                <div class="metric-value">{int(metrics['n_clusters'])}</div>
                <div class="metric-sub">{algorithm.split()[0]} Partition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Silhouette Score</div>
                <div class="metric-value">{metrics['silhouette']:.3f}</div>
                <div class="metric-sub">Davies-Bouldin: {metrics['davies_bouldin']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi5:
        recs_count = len(data["recs_df"])
        merger_count = len(data["recs_df"][data["recs_df"]["suggested_action"] == "Merge Candidate"])
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Redundancy Pairs</div>
                <div class="metric-value">{recs_count}</div>
                <div class="metric-sub">{merger_count} Merge Candidates</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN TABBED INTERFACE
    # -------------------------------------------------------------------------
    tab_map, tab_heat, tab_pca, tab_bench, tab_recs, tab_arch = st.tabs(
        [
            "🗺️ Cluster Space",
            "🔥 Overlap Matrix",
            "📊 PCA Spectrum & Loadings",
            "🔬 Ablation Benchmark",
            "📋 Interoperability Actions",
            "📐 System Architecture",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: CLUSTER SPACE (2D & 3D)
    # -------------------------------------------------------------------------
    with tab_map:
        st.markdown("#### 🗺️ Platform Distribution in Reduced Feature Space")
        st.caption(
            "Visualizing government platforms clustered by service-usage features. "
            "Adjacent platforms exhibit high shared service capabilities and redundant citizen workflows."
        )

        cluster_df = data["cluster_df"].copy()
        cluster_df["dynamic_cluster"] = [
            f"Cluster {c}" if c != -1 else "Noise (Outlier)" for c in active_labels
        ]
        cluster_df["PC1"] = X_pca_dyn[:, 0]
        cluster_df["PC2"] = X_pca_dyn[:, 1]
        cluster_df["PC3"] = X_pca_dyn[:, 2]

        color_discrete = px.colors.qualitative.Bold

        if proj_dimension == "3D Interactive Space" and proj_type.startswith("PCA"):
            fig_cluster = px.scatter_3d(
                cluster_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="dynamic_cluster",
                hover_name="platform_name",
                hover_data={"tier": True, "PC1": False, "PC2": False, "PC3": False, "dynamic_cluster": True},
                color_discrete_sequence=color_discrete,
                title=f"3D PCA Manifold (PC1: {pca_full.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca_full.explained_variance_ratio_[1]*100:.1f}%, PC3: {pca_full.explained_variance_ratio_[2]*100:.1f}% var)",
                height=650,
            )
            fig_cluster.update_traces(
                marker=dict(size=7, opacity=0.88, line=dict(width=0.5, color="#0f172a"))
            )
            fig_cluster.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis_title=f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)",
                    zaxis_title=f"PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}%)",
                ),
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        else:
            if proj_type.startswith("PCA"):
                x_col, y_col = "PC1", "PC2"
                x_label = f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% variance)"
                y_label = f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% variance)"
            else:
                x_col, y_col = "umap_1", "umap_2"
                x_label, y_label = "UMAP Dimension 1", "UMAP Dimension 2"

            fig_cluster = px.scatter(
                cluster_df,
                x=x_col,
                y=y_col,
                color="dynamic_cluster",
                hover_name="platform_name",
                hover_data={"tier": True, x_col: False, y_col: False, "dynamic_cluster": True},
                color_discrete_sequence=color_discrete,
                title=f"Platform Clusters in {proj_type} Space (Algorithm: {algorithm}, k={metrics['n_clusters']:.0f})",
                height=560,
                template="plotly_white",
            )
            fig_cluster.update_traces(
                marker=dict(size=14, line=dict(width=1.2, color="#0f172a"), opacity=0.9)
            )
            fig_cluster.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=30, l=40, r=40),
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        # Cluster Breakdown Inspector
        st.markdown("---")
        st.markdown("##### 🔍 Cluster Deep-Dive Directory")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            sel_cluster = st.selectbox(
                "Select Cluster to Inspect:",
                sorted(cluster_df["dynamic_cluster"].unique()),
            )
            members = cluster_df[cluster_df["dynamic_cluster"] == sel_cluster]
            st.dataframe(
                members[["platform_name", "tier"]].reset_index(drop=True),
                use_container_width=True,
                height=250,
            )

        with c_right:
            # Show category coverage distribution for this selected cluster
            matched_pids = members["platform_id"].tolist()
            sub_platforms = platforms_df[platforms_df["platform_id"].isin(matched_pids)]
            cov_cols = [f"coverage_{c}" for c in CATEGORIES if f"coverage_{c}" in sub_platforms.columns]
            if cov_cols:
                mean_cov = sub_platforms[cov_cols].mean().reset_index()
                mean_cov.columns = ["Category", "Mean Coverage"]
                mean_cov["Category"] = mean_cov["Category"].str.replace("coverage_", "").str.replace("_", " ").str.title()
                mean_cov = mean_cov.sort_values(by="Mean Coverage", ascending=False).head(6)

                fig_cat = px.bar(
                    mean_cov,
                    x="Mean Coverage",
                    y="Category",
                    orientation="h",
                    title=f"Dominant Functional Categories for {sel_cluster}",
                    color="Mean Coverage",
                    color_continuous_scale="Purples",
                    height=250,
                    template="plotly_white",
                )
                fig_cat.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_cat, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: OVERLAP MATRIX & COMPARATOR
    # -------------------------------------------------------------------------
    with tab_heat:
        st.markdown("#### 🔥 Pairwise Platform Service Overlap Heatmap")
        st.caption(
            "Blended similarity score: $0.65 \\times \\text{CosineSim}(\\text{Categories}) + 0.35 \\times \\text{Jaccard}(\\text{Active Services})$. "
            "Values approaching **1.0 (bright yellow)** identify high redundancy."
        )

        overlap_df = data["overlap_df"]

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            heat_filter = st.selectbox(
                "Filter Heatmap View:",
                ["Full National Matrix (All Platforms)", "Focus on Selected Cluster", "Top 20 Platforms by Overlap"],
            )
        with col_h2:
            heat_palette = st.selectbox(
                "Heatmap Color Palette:",
                ["Viridis", "Plasma", "Inferno", "YlOrRd", "Blues"],
                index=0,
            )

        if heat_filter == "Focus on Selected Cluster":
            target_cluster = st.selectbox(
                "Choose Cluster:",
                sorted(cluster_df["dynamic_cluster"].unique()),
                key="heat_cluster_sel",
            )
            cluster_pnames = cluster_df[cluster_df["dynamic_cluster"] == target_cluster]["platform_name"].tolist()
            cluster_pnames = [p for p in cluster_pnames if p in overlap_df.index]
            disp_overlap = overlap_df.loc[cluster_pnames, cluster_pnames] if cluster_pnames else overlap_df
        elif heat_filter == "Top 20 Platforms by Overlap":
            # Extract top 20 platforms with highest non-diagonal average overlap
            avg_overlap = overlap_df.replace(1.0, np.nan).mean(axis=1).sort_values(ascending=False).head(20).index.tolist()
            disp_overlap = overlap_df.loc[avg_overlap, avg_overlap]
        else:
            disp_overlap = overlap_df

        fig_heat = px.imshow(
            disp_overlap,
            labels=dict(x="Platform B", y="Platform A", color="Overlap"),
            x=disp_overlap.columns,
            y=disp_overlap.index,
            color_continuous_scale=heat_palette,
            zmin=0.0,
            zmax=1.0,
            aspect="auto",
            height=620,
        )
        fig_heat.update_layout(margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Platform-to-Platform Comparison Tool
        st.markdown("---")
        st.markdown("##### ⚖️ Head-to-Head Platform Comparator")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            p_a = st.selectbox("Select Platform A:", platforms_df["platform_name"].tolist(), index=0)
        with comp_col2:
            p_b = st.selectbox("Select Platform B:", platforms_df["platform_name"].tolist(), index=min(1, len(platforms_df)-1))

        if p_a and p_b:
            pair_score = float(overlap_df.loc[p_a, p_b]) if (p_a in overlap_df.index and p_b in overlap_df.columns) else 0.0

            action_badge = (
                '<span class="badge-merge">Merge Candidate</span>' if pair_score >= 0.75
                else '<span class="badge-gateway">Shared API Gateway</span>' if pair_score >= 0.55
                else '<span class="badge-sso">SSO & Identity Hub</span>' if pair_score >= 0.40
                else '<span class="badge-coexist">Federated Coexistence</span>'
            )

            res_c1, res_c2 = st.columns([1, 2])
            with res_c1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-top:14px;">
                        <div class="metric-title">Pairwise Overlap Score</div>
                        <div class="metric-value">{pair_score*100:.1f}%</div>
                        <div style="margin-top:6px;">{action_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with res_c2:
                # Compare category coverages
                p_a_row = platforms_df[platforms_df["platform_name"] == p_a].iloc[0]
                p_b_row = platforms_df[platforms_df["platform_name"] == p_b].iloc[0]
                cats = [c for c in CATEGORIES if f"coverage_{c}" in platforms_df.columns]
                comp_data = []
                for c in cats:
                    val_a = float(p_a_row.get(f"coverage_{c}", 0.0) or 0.0)
                    val_b = float(p_b_row.get(f"coverage_{c}", 0.0) or 0.0)
                    if val_a > 0.15 or val_b > 0.15:
                        comp_data.append({"Category": c.replace("_", " ").title(), p_a: val_a, p_b: val_b})

                if comp_data:
                    comp_df = pd.DataFrame(comp_data).melt(id_vars="Category", var_name="Platform", value_name="Coverage")
                    fig_radar = px.bar(
                        comp_df,
                        x="Category",
                        y="Coverage",
                        color="Platform",
                        barmode="group",
                        title=f"Service Category Alignment: {p_a} vs {p_b}",
                        height=280,
                        template="plotly_white",
                    )
                    fig_radar.update_layout(margin=dict(t=40, b=20, l=10, r=10))
                    st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: PCA VARIANCE SPECTRUM & LOADINGS
    # -------------------------------------------------------------------------
    with tab_pca:
        st.markdown("#### 📊 PCA Dimensionality Reduction Spectrum & Feature Drivers")
        st.caption(
            "SIH26129 Step 2: Unsupervised feature reduction preserves service-overlap variance while pruning uncorrelated noise. "
            "Feature loadings explain the primary latent semantic axes of government functionality."
        )

        p_left, p_right = st.columns([1, 1])

        scree_df = data["scree_df"]

        with p_left:
            st.markdown("##### Scree Plot (Cumulative Variance)")
            fig_scree = go.Figure()

            max_plot_comps = min(22, len(scree_df))
            fig_scree.add_trace(
                go.Bar(
                    x=scree_df["component_label"][:max_plot_comps],
                    y=scree_df["explained_variance_ratio"][:max_plot_comps] * 100,
                    name="Individual Var %",
                    marker_color="#4f46e5",
                    opacity=0.85,
                )
            )
            fig_scree.add_trace(
                go.Scatter(
                    x=scree_df["component_label"][:max_plot_comps],
                    y=scree_df["cumulative_variance_ratio"][:max_plot_comps] * 100,
                    name="Cumulative Var %",
                    mode="lines+markers",
                    marker=dict(size=6, color="#f43f5e"),
                    line=dict(width=2.5, color="#f43f5e"),
                    yaxis="y2",
                )
            )
            # Add target variance line
            fig_scree.add_hline(
                y=pca_variance * 100,
                yref="y2",
                line_dash="dot",
                line_color="#10b981",
                annotation_text=f"Target: {pca_variance*100:.0f}%",
                annotation_position="bottom right",
            )

            fig_scree.update_layout(
                template="plotly_white",
                height=460,
                yaxis=dict(title="Individual Variance (%)", showgrid=True, gridcolor="#f1f5f9"),
                yaxis2=dict(
                    title="Cumulative Variance (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    showgrid=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=30, l=40, r=40),
            )
            st.plotly_chart(fig_scree, use_container_width=True)

        with p_right:
            st.markdown("##### Feature Loadings (Explainability)")
            comp_selected = st.selectbox(
                "Select Principal Component to Inspect:",
                [f"PC{i+1}" for i in range(min(12, dyn_n_comps))],
                index=0,
            )
            comp_num = int(comp_selected.replace("PC", ""))
            loadings_df = data["loadings_df"]
            sub_load = loadings_df[loadings_df["component"] == comp_num].copy()

            fig_bar = px.bar(
                sub_load,
                x="loading_weight",
                y="feature_name",
                orientation="h",
                color="loading_weight",
                color_continuous_scale="RdBu",
                title=f"Highest Magnitude Drivers for {comp_selected}",
                height=460,
                template="plotly_white",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=50, b=30, l=40, r=20),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: ABLATION BENCHMARK
    # -------------------------------------------------------------------------
    with tab_bench:
        st.markdown("#### 🔬 Empirical Representation & Algorithm Benchmark")
        st.caption(
            "Ground-truth external validation comparing Raw Unreduced features (149D) against PCA reduced features (various dimensions) and UMAP. "
            "Evaluated using Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Random Seed Stability."
        )

        bench_df = data["benchmark_df"].copy()

        # Grouped bar chart comparing ARI and Stability across representations
        fig_bench = px.bar(
            bench_df,
            x="representation",
            y="adjusted_rand_index",
            color="algorithm",
            barmode="group",
            title="Ground Truth Alignment (Adjusted Rand Index) across Dimensionality Representations",
            labels={"adjusted_rand_index": "Adjusted Rand Index (ARI)", "representation": "Representation"},
            height=380,
            template="plotly_white",
            color_discrete_sequence=["#4f46e5", "#06b6d4", "#f59e0b"],
        )
        fig_bench.update_layout(margin=dict(t=40, b=40, l=30, r=20), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_bench, use_container_width=True)

        st.markdown("##### Detailed Metric Matrix")
        styled_bench = bench_df.sort_values(by="adjusted_rand_index", ascending=False).reset_index(drop=True)
        try:
            display_bench = styled_bench.style.background_gradient(
                subset=["adjusted_rand_index", "normalized_mutual_info", "silhouette"],
                cmap="YlGn",
            )
        except Exception:
            display_bench = styled_bench

        st.dataframe(
            display_bench,
            use_container_width=True,
            height=320,
        )

        st.success(
            "💡 **SIH26129 Empirical Takeaway**: High-dimensional raw service features suffer from the curse of dimensionality, "
            "causing clustering stability across random seeds to degrade to ~0.71. Applying PCA dimensionality reduction preserves 92% "
            "relevant variance, boosts clustering separation, and achieves perfect **ARI = 1.000** ground truth alignment with 100% stability.",
            icon="🏆",
        )

    # -------------------------------------------------------------------------
    # TAB 5: INTEROPERABILITY ACTIONS (RECOMMENDATIONS)
    # -------------------------------------------------------------------------
    with tab_recs:
        st.markdown("#### 📋 Prioritized Interoperability Policy Recommendations")
        st.caption(
            "SIH26129 Step 3: Algorithmic recommendations for digital governance rationalization. "
            "Identifies specific consolidation actions based on pairwise overlap thresholds."
        )

        recs_df = data["recs_df"].copy()

        # Action Summary Badges
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        with r_c1:
            m_cnt = len(recs_df[recs_df["suggested_action"] == "Merge Candidate"])
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 4px solid #ef4444;">
                    <div class="metric-title">Merge Candidates</div>
                    <div class="metric-value" style="color:#ef4444;">{m_cnt}</div>
                    <div class="metric-sub" style="color:#64748b;">Overlap ≥ 75%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c2:
            g_cnt = len(recs_df[recs_df["suggested_action"] == "Shared API Gateway"])
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                    <div class="metric-title">Shared API Gateways</div>
                    <div class="metric-value" style="color:#f59e0b;">{g_cnt}</div>
                    <div class="metric-sub" style="color:#64748b;">Overlap 55% - 74%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c3:
            s_cnt = len(recs_df[recs_df["suggested_action"] == "SSO & Identity Unification"])
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 4px solid #3b82f6;">
                    <div class="metric-title">SSO & Auth Sharing</div>
                    <div class="metric-value" style="color:#3b82f6;">{s_cnt}</div>
                    <div class="metric-sub" style="color:#64748b;">Overlap 40% - 54%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c4:
            c_cnt = len(recs_df[recs_df["suggested_action"] == "Federated Coexistence"])
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 4px solid #10b981;">
                    <div class="metric-title">Federated Coexistence</div>
                    <div class="metric-value" style="color:#10b981;">{c_cnt}</div>
                    <div class="metric-sub" style="color:#64748b;">Distinct Niche</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Filters
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            act_filter = st.multiselect(
                "Filter Action:",
                options=recs_df["suggested_action"].unique().tolist(),
                default=recs_df["suggested_action"].unique().tolist(),
            )
        with f_col2:
            min_score = st.slider("Min Overlap %:", 40.0, 95.0, 50.0, 5.0)
        with f_col3:
            search_query = st.text_input("Search Platform Name:", placeholder="e.g. Kisan, Bhulekh, Pension...")

        filtered = recs_df[
            (recs_df["suggested_action"].isin(act_filter))
            & (recs_df["overlap_pct"] >= min_score)
        ]
        if search_query:
            q = search_query.lower()
            filtered = filtered[
                filtered["platform_a"].str.lower().str.contains(q)
                | filtered["platform_b"].str.lower().str.contains(q)
            ]

        st.dataframe(
            filtered[
                [
                    "platform_a",
                    "platform_b",
                    "overlap_pct",
                    "suggested_action",
                    "shared_services",
                    "action_details",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        csv_download = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Policy Recommendations (CSV)",
            data=csv_download,
            file_name="sih26129_interoperability_recommendations.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # TAB 6: SYSTEM ARCHITECTURE & METHODOLOGY
    # -------------------------------------------------------------------------
    with tab_arch:
        st.markdown("#### 📐 SIH26129 System Architecture & Mathematical Foundations")
        st.markdown(
            """
            This platform solves **Problem SIH26129**: Disconnected government digital portals often duplicate services,
            resulting in fragmented citizen experiences and redundant infrastructure expenditure.
            """
        )

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown(
                """
                ##### 1. Multi-Block Feature Ingestion (149 Dimensions)
                Raw platforms are compiled across 7 heterogeneous information blocks:
                * **Category Coverage** (15 dims): Proportion of services offered across vital stats, land records, agriculture, health, etc.
                * **Granular Service Flags** (22 dims): Binary presence of critical citizen workflows (e.g., RoR 7/12, ABHA, PM-KISAN).
                * **Document Requirements** (8 dims): Authentication proofs required (Aadhaar, Land Deed, Ration Card).
                * **Authentication Protocols** (5 dims): DigiLocker, Biometric, Mobile OTP, Username/Password.
                * **API Capabilities** (4 dims): REST endpoints, Webhooks, Gov SDK compatibility.
                * **User Demographics** (5 dims): Rural, Urban, Senior Citizen, Youth percentages.
                * **Operational Usage Metrics** (10 dims): Monthly requests, latency, failure rates, grievance rates.
                * **TF-IDF Semantic Text** (80 dims): Unsupervised tokenization of service descriptions.
                """
            )

        with arch_col2:
            st.markdown(
                """
                ##### 2. Dimensionality Reduction & Overlap Metrics
                * **Variance Retention Criterion**:
                $$\\sum_{i=1}^d \\lambda_i \\Big/ \\sum_{j=1}^D \\lambda_j \\ge \\tau \\quad (\\tau = 0.92)$$
                * **Blended Interoperability Overlap Score**:
                $$\\text{Overlap}(P_A, P_B) = \\alpha \\cdot \\frac{v_A \\cdot v_B}{\\|v_A\\| \\|v_B\\|} + (1-\\alpha) \\cdot \\frac{|S_A \\cap S_B|}{|S_A \\cup S_B|}$$
                where $\\alpha = 0.65$ balances continuous functional coverage with discrete active citizen services.
                * **Decision Boundary**:
                  * Overlap $\\ge 75\\%$: **Merge Candidate**
                  * Overlap $55\\% - 74\\%$: **Shared API Gateway**
                  * Overlap $40\\% - 54\\%$: **SSO & Identity Unification**
                  * Overlap $< 40\\%$: **Federated Coexistence**
                """
            )


if __name__ == "__main__":
    main()
