"""Tools for Document-Term Matrix (DTM) construction and analysis."""

from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns DTM tools initialized with the given session."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        max_features: Optional[int] = None,
        stop_words: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from document text using CountVectorizer.

        Computes count-based term frequencies across the session dataframe's text column,
        populates session.feature_matrix and session.feature_names, sets category labels,
        stores the fitted vectorizer in session.artifacts['count_vectorizer'], and calculates DTM sparsity.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams to be extracted, e.g. [1, 1].
            min_df: Minimum document frequency for terms (int count or float proportion in [0.0, 1.0]).
            max_df: Maximum document frequency for terms (int count or float proportion in [0.0, 1.0]).
            max_features: Maximum number of vocabulary features to retain.
            stop_words: Language stop words parameter to pass to CountVectorizer (e.g. 'english' or None).

        Returns:
            A dictionary containing result_id, vocabulary_size, n_documents, non_zero, total_elements,
            sparsity_pct, and feature_names sample.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe must be loaded and contain a 'text' column before building DTM."}

        # Guard against integer 1 in max_df
        if isinstance(max_df, int) and max_df == 1:
            max_df = 1.0

        # Convert list to tuple for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        # Initialize vectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=stop_words,
        )

        texts = session.dataframe["text"].fillna("").astype(str).tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # Store in session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Sync categories / labels if available
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].values)

        # Sparsity calculation as defined in spec
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        res_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": res_id,
            "vocabulary_size": n_cols,
            "n_documents": n_rows,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "sample_feature_names": feature_names[:10],
        }

        stored_args = {
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
            "stop_words": stop_words,
        }
        session.store_result("build_dtm_tool", stored_args, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the global document-term matrix.

        Args:
            top_n: Number of top frequent terms to return in summary (default 20).

        Returns:
            A dictionary with result_id, total_terms, and top_term_frequencies list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is not loaded. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = list(session.feature_names)

        # Sum terms across all rows (documents)
        term_sums = np.array(X.sum(axis=0)).ravel()

        df_frequencies = pd.DataFrame({
            "term": feature_names,
            "frequency": term_sums
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        top_terms = df_frequencies.head(top_n).to_dict(orient="records")

        res_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": res_id,
            "total_terms": len(feature_names),
            "top_term_frequencies": top_terms,
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=df_frequencies)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Renders a raw positional slice heatmap of the global document-term matrix.

        Slices the top-left corner (first n_documents rows, first n_terms columns) of the existing matrix
        without any sorting or re-ordering. Sets a seaborn heatmap on session.pending_figure.

        Args:
            n_terms: Number of terms (columns) to slice from the vocabulary order.
            n_documents: Number of documents (rows) to slice from dataset order.

        Returns:
            A dictionary containing result_id, sliced_shape, terms_included, documents_included,
            and matrix_slice (as a nested Python list).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is not loaded. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = list(session.feature_names)

        n_rows, n_cols = X.shape
        actual_docs = min(n_documents, n_rows)
        actual_terms = min(n_terms, n_cols)

        # Raw positional slice matching Master's [0:actual_docs, 0:actual_terms]
        X_slice = X[0:actual_docs, 0:actual_terms].toarray()
        slice_terms = feature_names[0:actual_terms]
        doc_labels = [f"Doc_{i}" for i in range(actual_docs)]

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            X_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=slice_terms,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"DTM Slice Heatmap ({actual_docs} docs x {actual_terms} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        fig.tight_layout()

        # Set on session.pending_figure (no show or savefig)
        session.pending_figure = fig

        matrix_list = X_slice.tolist()

        res_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": res_id,
            "sliced_shape": [actual_docs, actual_terms],
            "terms_included": slice_terms,
            "documents_included": doc_labels,
            "matrix_slice": matrix_list,
        }

        df_report = pd.DataFrame(X_slice, columns=slice_terms, index=doc_labels)
        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary, full_report=df_report)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
