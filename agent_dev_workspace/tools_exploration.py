"""Tools for exploratory feature correlation analysis and document cosine similarity."""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns exploratory analysis tools initialized with the given session."""

    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents by their row indices in session.feature_matrix.

        Handles empty/zero vectors cleanly by returning 0.0 similarity instead of NaN.

        Args:
            doc_index_1: First document index (0-based).
            doc_index_2: Second document index (0-based).

        Returns:
            A dictionary with result_id, doc_index_1, doc_index_2, and similarity score.
        """
        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is not loaded. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_docs = X.shape[0]

        if not (0 <= doc_index_1 < n_docs and 0 <= doc_index_2 < n_docs):
            return {"error": f"Document indices must be between 0 and {n_docs - 1}."}

        # Extract 1D row vectors
        vec1 = X[doc_index_1]
        vec2 = X[doc_index_2]

        # Handle zero vector edge cases cleanly (empty text documents)
        norm1 = np.sqrt(vec1.power(2).sum())
        norm2 = np.sqrt(vec2.power(2).sum())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_mat = cosine_similarity(vec1, vec2)
            sim_val = float(sim_mat[0, 0])
            sim_val = 0.0 if np.isnan(sim_val) else sim_val

        res_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": res_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": round(sim_val, 4),
        }

        session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n_terms: int = 20) -> Dict[str, Any]:
        """Calculates Pearson correlation heatmap across top-variance terms in global DTM.

        Selects top_n_terms highest variance terms from session.feature_matrix, computes their
        term-vs-term Pearson correlation matrix, sets an annotated heatmap figure on
        session.pending_figure, and returns the correlation matrix in the summary.

        Args:
            top_n_terms: Number of top-variance terms to correlate (default 20).

        Returns:
            A dictionary with result_id, terms_included, correlation_matrix (as nested list),
            and matrix_shape.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is not loaded. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # 1. Compute term variances across all documents (sparse E[X^2] - (E[X])^2)
        x_mean = np.array(X.mean(axis=0)).ravel()
        x_mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        variances = x_mean_sq - (x_mean ** 2)

        # 2. Slice top_n_terms by variance descending
        actual_n = min(top_n_terms, X.shape[1])
        top_var_indices = np.argsort(variances)[::-1][:actual_n]

        sub_matrix = X[:, top_var_indices].toarray()
        sub_feature_names = feature_names[top_var_indices].tolist()

        # 3. Compute Pearson correlation matrix across term columns (shape: actual_n x actual_n)
        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)
        # Handle single term edge case or NaNs from constant columns
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 4. Build annotated heatmap figure and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=sub_feature_names,
            yticklabels=sub_feature_names,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_n} Variance Terms)")
        fig.tight_layout()

        # Set figure on session.pending_figure (no show or savefig)
        session.pending_figure = fig

        corr_list = corr_matrix.tolist()

        res_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": res_id,
            "terms_included": sub_feature_names,
            "correlation_matrix": [[round(v, 4) for v in row] for row in corr_list],
            "matrix_shape": [actual_n, actual_n],
        }

        df_report = pd.DataFrame(corr_matrix, columns=sub_feature_names, index=sub_feature_names)
        session.store_result("feature_correlation_matrix_tool", {"top_n_terms": top_n_terms}, summary, full_report=df_report)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
