"""Tools for dimensionality reduction and label encoding."""

from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import umap
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns dimensionality reduction tools initialized with the given session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        max_terms: int = 1500
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensionality using PCA, t-SNE, or UMAP.

        Computes 2D/low-dimensional coordinates from session.feature_matrix and builds a scatter plot
        colored by document category, stored on session.pending_figure.

        Args:
            method: Dimensionality reduction technique ("pca", "tsne", or "umap").
            n_components: Target output dimensions (default 2).
            random_state: Random seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors for UMAP.
            max_terms: Maximum number of highest-variance terms to retain before dense conversion (default 1500).

        Returns:
            A dictionary with result_id, method, n_components, explained_variance_ratio (if PCA),
            and sample 2D coordinates.
        """
        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is not loaded. Run build_dtm_tool first."}

        X_sparse = session.feature_matrix

        # Slice to top max_terms by variance before converting to dense array if matrix has more columns
        if X_sparse.shape[1] > max_terms:
            x_mean = np.array(X_sparse.mean(axis=0)).ravel()
            x_mean_sq = np.array(X_sparse.power(2).mean(axis=0)).ravel()
            variances = x_mean_sq - (x_mean ** 2)
            top_var_indices = np.argsort(variances)[::-1][:max_terms]
            X = X_sparse[:, top_var_indices].toarray()
            terms_used = max_terms
        else:
            X = X_sparse.toarray() if hasattr(X_sparse, "toarray") else X_sparse
            terms_used = X.shape[1]

        # Execute selected reduction algorithm
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            effective_perplexity = min(perplexity, max(1.0, float(X.shape[0] - 1)))
            reducer = TSNE(n_components=n_components, perplexity=effective_perplexity, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance = None
        elif method == "umap":
            effective_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
            reducer = umap.UMAP(n_components=n_components, n_neighbors=effective_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance = None
        else:
            return {"error": f"Unknown method '{method}'."}

        # Retrieve category labels for coloring
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()
        elif labels is None:
            labels = ["unlabeled"] * X.shape[0]

        # Construct DataFrame for coordinates and plotting
        coord_cols = [f"Dim_{i+1}" for i in range(n_components)]
        df_coords = pd.DataFrame(coords, columns=coord_cols)
        df_coords["category"] = labels

        # Build scatter plot and assign to session.pending_figure (no savefig or show)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_coords,
            x=coord_cols[0],
            y=coord_cols[1] if n_components > 1 else coord_cols[0],
            hue="category",
            palette="tab10",
            ax=ax,
            alpha=0.8
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel(coord_cols[0])
        ax.set_ylabel(coord_cols[1] if n_components > 1 else coord_cols[0])
        fig.tight_layout()

        # Set plot on session.pending_figure
        session.pending_figure = fig

        res_id = session.next_result_id("reduce_dimensions")

        # Compact summary with top 5 sample rows
        sample_coords = df_coords.head(5).to_dict(orient="records")

        summary = {
            "result_id": res_id,
            "method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "max_terms": max_terms,
            "terms_used": terms_used,
            "total_documents": len(df_coords),
            "sample_coordinates": sample_coords,
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = [round(v, 4) for v in explained_variance]

        # Store complete coordinates in session.artifacts and full_report
        artifact_key = f"reduced_coords_{method}"
        session.artifacts[artifact_key] = coords

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "max_terms": max_terms
        }, summary, full_report=df_coords)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes session.dataframe['category_name'] into a binary indicator matrix (1 column per class).

        Stores the binarized matrix in session.artifacts['binarized_labels'] and sorted class names
        in session.artifacts['binarized_label_classes'].

        Returns:
            A summary dictionary containing result_id, n_documents, n_classes, classes, and example_rows.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        if "category_name" not in df.columns:
            return {"error": "session.dataframe does not contain a 'category_name' column."}

        raw_labels = df["category_name"].astype(str).values.reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        encoded_matrix = encoder.fit_transform(raw_labels).astype(int)
        classes = [str(c) for c in encoder.categories_[0].tolist()]

        session.artifacts["binarized_labels"] = encoded_matrix
        session.artifacts["binarized_label_classes"] = classes

        example_rows = encoded_matrix[:3].tolist() if len(encoded_matrix) >= 3 else encoded_matrix.tolist()

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "n_documents": len(df),
            "n_classes": len(classes),
            "classes": classes,
            "example_rows": example_rows,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
