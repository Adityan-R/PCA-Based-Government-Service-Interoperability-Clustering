"""Overlap scoring and policy recommendation engine.

SIH26129: Computes pairwise platform service overlap matrices and produces
prioritized interoperability action recommendations (API gateway, SSO, Merger).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_gen import CATEGORIES, GRANULAR_SERVICE_FLAGS, load_config


def compute_pairwise_overlap_matrix(
    df_platforms: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Compute pairwise blended service overlap score matrix in range [0, 1].

    Blends cosine similarity of coverage vectors with Jaccard index of active services.
    """
    platform_ids = df_platforms["platform_id"].tolist()
    platform_names = df_platforms["platform_name"].tolist()

    # Extract coverage and service flag columns
    feature_cols = [f"coverage_{c}" for c in CATEGORIES if f"coverage_{c}" in df_platforms.columns]
    feature_cols += [flg for flg in GRANULAR_SERVICE_FLAGS if flg in df_platforms.columns]

    X_service = df_platforms[feature_cols].fillna(0.0).to_numpy(dtype=float)

    # 1. Cosine similarity
    cos_sim = cosine_similarity(X_service)

    # 2. Binary Jaccard similarity
    binary_mask = (X_service > 0.25).astype(int)
    n = len(df_platforms)
    jaccard_sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                jaccard_sim[i, j] = 1.0
            else:
                inter = np.sum(np.bitwise_and(binary_mask[i], binary_mask[j]))
                union = np.sum(np.bitwise_or(binary_mask[i], binary_mask[j]))
                jaccard_sim[i, j] = inter / union if union > 0 else 0.0

    # Blended overlap score strictly clamped to [0.0, 1.0]
    blended_matrix = 0.65 * cos_sim + 0.35 * jaccard_sim
    np.fill_diagonal(blended_matrix, 1.0)
    blended_matrix = np.clip(blended_matrix, 0.0, 1.0)

    return blended_matrix, platform_ids, platform_names


def extract_shared_services(
    row1: pd.Series, row2: pd.Series, top_k: int = 3
) -> str:
    """Identify top overlapping functional categories and service flags between two platforms."""
    shared = []
    for c in CATEGORIES:
        col = f"coverage_{c}"
        if col in row1 and col in row2:
            v1 = float(row1.get(col, 0.0) or 0.0)
            v2 = float(row2.get(col, 0.0) or 0.0)
            if v1 > 0.30 and v2 > 0.30:
                shared.append(c.replace("_", " ").title())

    if not shared:
        for flg in GRANULAR_SERVICE_FLAGS:
            if flg in row1 and flg in row2:
                if int(row1.get(flg, 0) or 0) == 1 and int(row2.get(flg, 0) or 0) == 1:
                    shared.append(flg.replace("flag_", "").replace("_", " ").title())

    if not shared:
        return "General Citizen Public Services"

    return ", ".join(shared[:top_k])


def generate_recommendations(
    df_platforms: pd.DataFrame,
    cluster_labels: np.ndarray,
    overlap_matrix: np.ndarray,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate ranked interoperability recommendations from cluster memberships and overlap scores."""
    if config is None:
        config = load_config()

    recommendations: List[Dict[str, Any]] = []
    n = len(df_platforms)

    for i in range(n):
        for j in range(i + 1, n):
            c_i = int(cluster_labels[i])
            c_j = int(cluster_labels[j])
            overlap_score = float(np.round(overlap_matrix[i, j], 4))

            # Focus primarily on pairs in the same cluster or high overlap cross-cluster
            same_cluster = bool(c_i == c_j and c_i != -1)

            if same_cluster or overlap_score >= 0.45:
                p1_name = df_platforms.iloc[i]["platform_name"]
                p2_name = df_platforms.iloc[j]["platform_name"]
                tier1 = df_platforms.iloc[i].get("tier", "State")
                tier2 = df_platforms.iloc[j].get("tier", "State")

                shared_text = extract_shared_services(df_platforms.iloc[i], df_platforms.iloc[j])

                # Determine suggested policy action
                if overlap_score >= 0.75:
                    action_type = "Merge Candidate"
                    action_desc = "High redundancy: Consolidate redundant digital portals into a single unified citizen interface."
                elif overlap_score >= 0.55:
                    action_type = "Shared API Gateway"
                    action_desc = "Substantial service overlap: Establish a unified REST API gateway & interoperable data exchange schema."
                elif overlap_score >= 0.40:
                    action_type = "SSO & Identity Unification"
                    action_desc = "Common citizen cohort: Implement unified Single Sign-On (SSO) with DigiLocker and Mobile OTP."
                else:
                    action_type = "Federated Coexistence"
                    action_desc = "Moderate synergy: Enable reciprocal deep linking and shared directory discovery."

                recommendations.append(
                    {
                        "platform_a": p1_name,
                        "platform_b": p2_name,
                        "tier_a": tier1,
                        "tier_b": tier2,
                        "cluster_id": c_i if same_cluster else -1,
                        "same_cluster": same_cluster,
                        "overlap_pct": float(np.round(overlap_score * 100, 2)),
                        "overlap_score": overlap_score,
                        "shared_services": shared_text,
                        "suggested_action": action_type,
                        "action_details": action_desc,
                    }
                )

    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df = rec_df.sort_values(by="overlap_score", ascending=False).reset_index(drop=True)

    # Save to data/processed/recommendations.csv
    proc_dir = Path(config["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(config["paths"]["recommendations_csv"], index=False)

    return rec_df


def run_overlap_pipeline(
    df_platforms: pd.DataFrame,
    cluster_labels: np.ndarray,
    config: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Execute overlap scoring and recommendation generation."""
    if config is None:
        config = load_config()

    overlap_mat, pids, pnames = compute_pairwise_overlap_matrix(df_platforms)

    # Save overlap matrix with headers
    overlap_df = pd.DataFrame(overlap_mat, index=pnames, columns=pnames)
    overlap_df.to_csv(config["paths"]["overlap_matrix_csv"])

    rec_df = generate_recommendations(df_platforms, cluster_labels, overlap_mat, config)
    return overlap_mat, rec_df


if __name__ == "__main__":
    cfg = load_config()
    platforms_df = pd.read_csv(cfg["paths"]["platforms_csv"])
    X_pca = np.load(cfg["paths"]["pca_features_npy"])

    from src.cluster import ClusterSuite

    suite = ClusterSuite(cfg)
    labels, _ = suite.run_kmeans(X_pca, k=cfg["clustering"]["default_k"])

    mat, recs = run_overlap_pipeline(platforms_df, labels, cfg)
    print(f"Overlap matrix computed: {mat.shape}, Range: [{mat.min():.3f}, {mat.max():.3f}]")
    print(f"Generated {len(recs)} policy recommendations.")
    print("\nTop 5 Interoperability Recommendations:")
    print(recs[["platform_a", "platform_b", "overlap_pct", "suggested_action", "shared_services"]].head(5))
