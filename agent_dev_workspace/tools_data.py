"""Tools for data loading, inspection, tokenization, missing/duplicate checks, sampling, and summary statistics."""

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns data management tools initialized with the given session."""

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks session.dataframe for missing or empty text values.

        Evaluates rows where the 'text' column is NaN, None, or consists only of whitespace / empty strings,
        mirroring the data quality checking step.

        Returns:
            A dictionary containing result_id, total_rows, missing_count, missing_indices,
            and missing_percentage.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        total_rows = len(df)

        # Identify missing or empty text values (nulls or whitespace-only strings)
        missing_mask = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
        missing_indices = df[missing_mask].index.tolist()
        missing_count = len(missing_indices)
        missing_pct = float(100.0 * missing_count / total_rows) if total_rows > 0 else 0.0

        res_id = session.next_result_id("check_missing")
        summary = {
            "result_id": res_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "missing_percentage": round(missing_pct, 4),
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate text entries in session.dataframe, and optionally drops them.

        Counts duplicate rows using default keep='first' (counts extra copies, e.g., 1 duplicate row).
        When dropping duplicates (drop=True), uses keep=False to drop ALL copies of duplicate rows,
        matching Master's drop_duplicates strategy.

        Args:
            drop: If True, mutates session.dataframe in place to remove all duplicate rows.

        Returns:
            A dictionary containing result_id, total_rows_before, duplicate_count,
            duplicate_indices, dropped, and total_rows_after.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        total_rows_before = len(df)

        # Identify duplicate rows based on default keep='first' (flags extra duplicate copies)
        dup_mask = df.duplicated(subset=["text"], keep="first")
        dup_indices = df[dup_mask].index.tolist()
        dup_count = len(dup_indices)

        if drop and dup_count > 0:
            session.dataframe = df.drop_duplicates(subset=["text"], keep=False).reset_index(drop=True)
            total_rows_after = len(session.dataframe)
            dropped = True
        else:
            total_rows_after = total_rows_before
            dropped = False

        res_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": res_id,
            "total_rows_before": total_rows_before,
            "duplicate_count": dup_count,
            "duplicate_indices": dup_indices,
            "dropped": dropped,
            "total_rows_after": total_rows_after,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text in session.dataframe into unigrams using nltk.word_tokenize after lowercasing.

        Stores list of token lists in session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Returns:
            A summary dictionary containing result_id, n_documents, total_tokens, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        tokenized_docs = []
        for text in df["text"]:
            if pd.isna(text) or text is None:
                tokenized_docs.append([])
            else:
                tokenized_docs.append(nltk.word_tokenize(str(text).lower()))

        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        total_tokens = sum(len(doc) for doc in tokenized_docs)
        sample_tokens = tokenized_docs[0] if len(tokenized_docs) > 0 else []

        res_id = session.next_result_id("tokenize")
        summary = {
            "result_id": res_id,
            "n_documents": len(df),
            "total_tokens": total_tokens,
            "sample_tokens": sample_tokens,
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsamples session.dataframe using pandas sample.

        Mutates session.dataframe in place and updates session labels/categories if category_name is present.

        Args:
            n: Number of samples to draw.
            random_state: Random seed for reproducibility.

        Returns:
            A summary dictionary containing result_id, total_rows_before, sampled_rows, sampled_indices, and n.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        total_rows_before = len(df)
        sampled_n = min(n, total_rows_before)

        sampled_df = df.sample(n=sampled_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        session.dataframe = sampled_df.reset_index(drop=True)

        if "category_name" in session.dataframe.columns and hasattr(session, "set_labels"):
            session.set_labels(session.dataframe["category_name"].values)

        res_id = session.next_result_id("sample_data")
        summary = {
            "result_id": res_id,
            "total_rows_before": total_rows_before,
            "sampled_rows": len(session.dataframe),
            "sampled_indices": sampled_indices,
            "n": n,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and folders in the specified workspace subdirectory.

        Args:
            subdirectory: Path relative to repository root (default: '.').

        Returns:
            A summary dictionary with result_id, subdirectory, files, and directories.
        """
        try:
            base_dir = os.path.abspath(".")
            target_path = os.path.abspath(os.path.join(base_dir, subdirectory))

            # Prevent directory traversal outside current working directory
            if not target_path.startswith(base_dir):
                return {"error": "Access denied: Path resolves outside current workspace."}

            entries = os.listdir(target_path)
            files = sorted([e for e in entries if os.path.isfile(os.path.join(target_path, e))])
            directories = sorted([e for e in entries if os.path.isdir(os.path.join(target_path, e))])

            res_id = session.next_result_id("list_files")
            summary = {
                "result_id": res_id,
                "subdirectory": subdirectory,
                "files": files,
                "directories": directories,
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary
        except Exception as e:
            return {"error": f"Failed to list directory '{subdirectory}': {str(e)}"}

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text character length stats overall and per category, and generates a box plot.

        Calculates text_length as df['text'].fillna("").apply(len). Prepares describe() standard stats
        (count, mean, std, min, 25%, 50%, 75%, max) overall and by category_name, and creates
        a seaborn boxplot set on session.pending_figure.

        Returns:
            A summary dictionary containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe.copy()
        if "text" not in df.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        overall_desc = df["text_length"].describe().to_dict()
        overall_stats = {k: float(v) if pd.notna(v) else None for k, v in overall_desc.items()}

        per_category_stats = {}
        if "category_name" in df.columns:
            for cat_name, group in df.groupby("category_name"):
                cat_desc = group["text_length"].describe().to_dict()
                per_category_stats[str(cat_name)] = {
                    k: float(v) if pd.notna(v) else None for k, v in cat_desc.items()
                }

        # Build Seaborn boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution")
        plt.tight_layout()

        session.pending_figure = fig

        res_id = session.next_result_id("describe_data")
        summary = {
            "result_id": res_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        text_max_chars: int = 200,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Peeks at the first n rows of session.dataframe, truncating long text for LLM inspectability.

        Args:
            n: Number of head rows to show (default: 5, capped at 20).
            text_max_chars: Maximum character length for text strings before truncation (default: 200).
            columns: Optional list of columns to include. If None, uses ['text', 'category_name'] if available.

        Returns:
            A summary dictionary containing result_id, total_rows, columns, and rows.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        total_rows = len(df)
        effective_n = min(n, 20)

        if columns is None:
            cols_to_use = [c for c in ["text", "category_name"] if c in df.columns]
            if not cols_to_use:
                cols_to_use = df.columns.tolist()
        else:
            cols_to_use = [c for c in columns if c in df.columns]

        head_df = df.head(effective_n)[cols_to_use].copy()

        rows = []
        for idx, row in head_df.iterrows():
            row_dict = {}
            for col in cols_to_use:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, str) and len(val) > text_max_chars:
                    cutoff = max(0, text_max_chars - 3)
                    row_dict[col] = val[:cutoff] + "..."
                else:
                    row_dict[col] = val
            rows.append(row_dict)

        res_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": res_id,
            "total_rows": total_rows,
            "columns": cols_to_use,
            "rows": rows,
        }

        session.store_result(
            "inspect_data_tool", {"n": n, "text_max_chars": text_max_chars}, summary
        )
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        list_files_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
