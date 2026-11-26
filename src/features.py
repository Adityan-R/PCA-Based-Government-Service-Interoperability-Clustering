"""Feature preprocessing and multi-block feature pipeline.

Combines category coverage, TF-IDF text features, document requirements,
auth methods, API capabilities, user segments, and scaled operational metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data_gen import (
    API_CAPABILITIES,
    AUTH_METHODS,
    CATEGORIES,
    DOCUMENTS,
    GRANULAR_SERVICE_FLAGS,
    METRICS,
    USER_SEGMENTS,
    load_config,
)


class PlatformFeaturePipeline:
    """Preprocesses tabular and text features from platform records."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
        f_cfg = config.get("features", {})

        self.tfidf_max_features = int(f_cfg.get("tfidf_max_features", 80))
        self.svd_components = int(f_cfg.get("svd_text_components", 20))
        self.scaling_method = f_cfg.get("scaling", "standard")

        self.imputer_numeric = SimpleImputer(strategy=f_cfg.get("imputation_strategy_numeric", "median"))
        self.scaler = StandardScaler() if self.scaling_method == "standard" else RobustScaler()

        self.tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.svd = TruncatedSVD(
            n_components=self.svd_components,
            random_state=int(config.get("seed", 42)),
        )

        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _extract_blocks(
        self, df: pd.DataFrame, fit: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and assemble all feature blocks with full tracking of column names."""
        n_samples = len(df)

        # 1. Category Coverage Block (15 dims)
        cov_cols = [f"coverage_{c}" for c in CATEGORIES if f"coverage_{c}" in df.columns]
        cov_data = df[cov_cols].to_numpy(dtype=float) if cov_cols else np.zeros((n_samples, len(CATEGORIES)))
        cov_names = [f"cov__{c}" for c in CATEGORIES]

        # 2. Granular Service Flags (22 dims)
        flag_cols = [f for f in GRANULAR_SERVICE_FLAGS if f in df.columns]
        flag_data = df[flag_cols].to_numpy(dtype=float) if flag_cols else np.zeros((n_samples, len(GRANULAR_SERVICE_FLAGS)))
        flag_names = [f"service__{f.replace('flag_', '')}" for f in GRANULAR_SERVICE_FLAGS]

        # 3. Document Requirement Flags (8 dims)
        doc_cols = [d for d in DOCUMENTS if d in df.columns]
        doc_data = df[doc_cols].to_numpy(dtype=float) if doc_cols else np.zeros((n_samples, len(DOCUMENTS)))
        doc_names = [f"doc__{d.replace('doc_', '')}" for d in DOCUMENTS]

        # 4. Auth Method Flags (5 dims)
        auth_cols = [a for a in AUTH_METHODS if a in df.columns]
        auth_data = df[auth_cols].to_numpy(dtype=float) if auth_cols else np.zeros((n_samples, len(AUTH_METHODS)))
        auth_names = [f"auth__{a.replace('auth_', '')}" for a in AUTH_METHODS]

        # 5. API Capabilities (4 dims)
        api_cols = [a for a in API_CAPABILITIES if a in df.columns]
        api_data = df[api_cols].to_numpy(dtype=float) if api_cols else np.zeros((n_samples, len(API_CAPABILITIES)))
        api_names = [f"api__{a.replace('api_', '')}" for a in API_CAPABILITIES]

        # 6. User Segments (5 dims)
        seg_cols = [s for s in USER_SEGMENTS if s in df.columns]
        seg_data = df[seg_cols].to_numpy(dtype=float) if seg_cols else np.zeros((n_samples, len(USER_SEGMENTS)))
        seg_names = [f"seg__{s.replace('user_seg_', '').replace('_pct', '')}" for s in USER_SEGMENTS]

        # 7. Operational Usage Metrics (10 dims)
        metric_cols = [m for m in METRICS if m in df.columns]
        metric_data = df[metric_cols].to_numpy(dtype=float) if metric_cols else np.zeros((n_samples, len(METRICS)))
        metric_names = [f"metric__{m.replace('metric_', '')}" for m in METRICS]

        # Combine tabular numeric block
        tabular_raw = np.hstack([cov_data, flag_data, doc_data, auth_data, api_data, seg_data, metric_data])
        tabular_names = cov_names + flag_names + doc_names + auth_names + api_names + seg_names + metric_names

        # Impute missing values in tabular block
        if fit:
            tabular_imputed = self.imputer_numeric.fit_transform(tabular_raw)
            tabular_scaled = self.scaler.fit_transform(tabular_imputed)
        else:
            tabular_imputed = self.imputer_numeric.transform(tabular_raw)
            tabular_scaled = self.scaler.transform(tabular_imputed)

        # 8. Text features from service descriptions (TF-IDF direct vocabulary representation)
        text_series = df.get("service_description_text", pd.Series([""] * n_samples)).fillna("").astype(str)
        if fit:
            tfidf_mat = self.tfidf.fit_transform(text_series)
        else:
            tfidf_mat = self.tfidf.transform(text_series)

        text_dense = tfidf_mat.toarray()
        feature_vocab = self.tfidf.get_feature_names_out()
        text_names = [f"tfidf__{w.replace(' ', '_')}" for w in feature_vocab]

        # Full feature matrix fusion
        X_full = np.hstack([tabular_scaled, text_dense])
        feature_names = tabular_names + text_names

        return X_full, feature_names

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """Fit transformers and return (X, feature_names, platform_ids, platform_names)."""
        X, feature_names = self._extract_blocks(df, fit=True)
        self.feature_names_ = feature_names
        self.is_fitted = True

        platform_ids = df["platform_id"].tolist() if "platform_id" in df.columns else [f"P_{i}" for i in range(len(df))]
        platform_names = df["platform_name"].tolist() if "platform_name" in df.columns else platform_ids

        return X, feature_names, platform_ids, platform_names

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Transform new dataframe using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")
        X, _ = self._extract_blocks(df, fit=False)
        platform_ids = df["platform_id"].tolist() if "platform_id" in df.columns else [f"P_{i}" for i in range(len(df))]
        platform_names = df["platform_name"].tolist() if "platform_name" in df.columns else platform_ids
        return X, platform_ids, platform_names


def process_and_save_features(
    df_platforms: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Preprocess platform features and save feature matrix to disk."""
    if config is None:
        config = load_config()

    pipeline = PlatformFeaturePipeline(config)
    X, feat_names, pids, pnames = pipeline.fit_transform(df_platforms)

    out_df = pd.DataFrame(X, columns=feat_names)
    out_df.insert(0, "platform_id", pids)
    out_df.insert(1, "platform_name", pnames)

    proc_dir = Path(config["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(config["paths"]["features_csv"], index=False)

    return X, feat_names, out_df


if __name__ == "__main__":
    cfg = load_config()
    raw_df = pd.read_csv(cfg["paths"]["platforms_csv"])
    X_mat, names, feat_df = process_and_save_features(raw_df, cfg)
    print(f"Feature matrix built: {X_mat.shape} (platforms x features)")
    print(f"First 10 features: {names[:10]}")
