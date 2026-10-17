"""Tools for feature selection using variance, Pearson correlation, and Spearman correlation."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns feature filtering tools initialized with the given session."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags near-constant terms by variance for removal using the global DTM.

        Calculates term-wise variance across session.feature_matrix and keeps terms with
        variance strictly greater than threshold (> threshold), matching VarianceThreshold behavior.

        Args:
            threshold: Variance cutoff threshold. Terms with variance > threshold are kept.

        Returns:
            A dictionary with result_id, total_terms, kept_count, removed_count,
            top_kept_terms, and top_removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is not loaded. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Efficient sparse column variance calculation: Var(X) = E[X^2] - (E[X])^2
        mean = np.array(X.mean(axis=0)).ravel()
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # Strict inequality matching sklearn VarianceThreshold (v > threshold)
        kept_mask = variances > threshold
        removed_mask = ~kept_mask

        kept_count = int(np.sum(kept_mask))
        removed_count = int(np.sum(removed_mask))

        # Full report DataFrame for visualize_result_tool (Term in 1st col, Score in 2nd col)
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Top terms for summary to avoid overwhelming LLM context on large vocabularies
        top_kept = df_report[df_report["variance"] > threshold]["term"].head(10).tolist()
        top_removed = df_report[df_report["variance"] <= threshold]["term"].head(10).tolist()

        res_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": res_id,
            "total_terms": len(feature_names),
            "kept_count": kept_count,
            "removed_count": removed_count,
            "top_kept_terms": top_kept,
            "top_removed_terms": top_removed,
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df_report)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Calculates Pearson correlation between term counts and a target class (one-vs-rest).

        Vectorized for high-dimensional sparse matrices.

        Args:
            target_class: The binary positive class name to evaluate correlation against.
            threshold: Minimum absolute Pearson r correlation score required to keep a feature (|r| >= threshold).

        Returns:
            A dictionary with result_id, target_class, total_terms, kept_count, removed_count,
            top_kept_terms, and top_removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "session.feature_matrix, session.feature_names, or session.labels is not loaded."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        y = (np.array(session.labels) == target_class).astype(float)

        # Vectorized Pearson r calculation for sparse matrix X against 1D vector y
        n = X.shape[0]
        y_mean = np.mean(y)
        y_std = np.std(y, ddof=0)

        if y_std == 0:
            pearson_rs = np.zeros(X.shape[1])
        else:
            x_mean = np.array(X.mean(axis=0)).ravel()
            x_mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            x_std = np.sqrt(np.maximum(0.0, x_mean_sq - (x_mean ** 2)))

            # Covariance(X, y) = E[X*y] - E[X]*E[y]
            # X.T @ y computes sum_i(x_ij * y_i) across documents efficiently
            xy_mean = np.array(X.T.dot(y)).ravel() / n
            cov_xy = xy_mean - (x_mean * y_mean)

            # Avoid division by zero for constant terms
            with np.errstate(divide="ignore", invalid="ignore"):
                pearson_rs = cov_xy / (x_std * y_std)
                pearson_rs = np.nan_to_num(pearson_rs, nan=0.0)

        kept_mask = np.abs(pearson_rs) >= threshold
        kept_count = int(np.sum(kept_mask))
        removed_count = int(len(pearson_rs) - kept_count)

        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_rs
        })
        df_report["abs_r"] = df_report["pearson_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        top_kept = df_report[df_report["pearson_r"].abs() >= threshold]["term"].head(10).tolist()
        top_removed = df_report[df_report["pearson_r"].abs() < threshold]["term"].head(10).tolist()

        res_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "total_terms": len(feature_names),
            "kept_count": kept_count,
            "removed_count": removed_count,
            "top_kept_terms": top_kept,
            "top_removed_terms": top_removed,
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=df_report)
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0, max_terms: int = 1500) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between term counts and a target class (one-vs-rest).

        Optionally restricts calculation to top max_terms by variance for scalability on large corpora.

        Args:
            target_class: The binary positive class name to evaluate correlation against.
            threshold: Minimum absolute Spearman correlation score required to keep a feature.
            max_terms: Maximum number of highest-variance terms to calculate Spearman correlation for (default 1500).

        Returns:
            A dictionary with result_id, target_class, total_terms, evaluated_terms, kept_count,
            removed_count, top_kept_terms, and top_removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "session.feature_matrix, session.feature_names, or session.labels is not loaded."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        y = (np.array(session.labels) == target_class).astype(float)

        # Pre-filter to top max_terms by variance if total terms exceeds max_terms
        if X.shape[1] > max_terms:
            x_mean = np.array(X.mean(axis=0)).ravel()
            x_mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = x_mean_sq - (x_mean ** 2)
            top_var_indices = np.argsort(variances)[::-1][:max_terms]
            X_sub = X[:, top_var_indices]
            sub_feature_names = feature_names[top_var_indices]
        else:
            X_sub = X
            sub_feature_names = feature_names

        dense_sub = X_sub.toarray()
        spearman_rs = []
        for col_idx in range(dense_sub.shape[1]):
            x_col = dense_sub[:, col_idx].astype(float)
            if np.std(x_col) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r_val, _ = stats.spearmanr(x_col, y)
                r = 0.0 if np.isnan(r_val) else float(r_val)
            spearman_rs.append(r)

        spearman_rs = np.array(spearman_rs)
        kept_mask = np.abs(spearman_rs) >= threshold
        kept_count = int(np.sum(kept_mask))
        removed_count = int(len(spearman_rs) - kept_count)

        df_report = pd.DataFrame({
            "term": sub_feature_names,
            "spearman_r": spearman_rs
        })
        df_report["abs_r"] = df_report["spearman_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        top_kept = df_report[df_report["spearman_r"].abs() >= threshold]["term"].head(10).tolist()
        top_removed = df_report[df_report["spearman_r"].abs() < threshold]["term"].head(10).tolist()

        res_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "total_terms": len(feature_names),
            "evaluated_terms": len(sub_feature_names),
            "kept_count": kept_count,
            "removed_count": removed_count,
            "top_kept_terms": top_kept,
            "top_removed_terms": top_removed,
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold, "max_terms": max_terms}, summary, full_report=df_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
