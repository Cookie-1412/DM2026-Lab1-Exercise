"""Tools for frequent pattern mining using PAMI algorithms on per-category text data."""

import os
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports for frequent pattern algorithms
try:
    from PAMI.frequentPattern.basic import FPGrowth
    from PAMI.frequentPattern.topk import FAE
    from PAMI.frequentPattern.maximal import MaxFPGrowth
except ImportError:
    pass


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns pattern mining tools initialized with the given session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10
    ) -> Dict[str, Any]:
        """Mines frequent word patterns for a specific document category using PAMI algorithms.

        Constructs a fresh per-category document-term matrix with English stop-words removed,
        filters terms using variance, tfidf, or term_frequency, converts the result into a
        transactional database, and runs fpgrowth, topk, or maxfpgrowth.

        Args:
            category_name: Target document category to mine.
            filtering_method: Filtering strategy ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support count required for "fpgrowth" and "maxfpgrowth".
            k: Number of highest-support patterns to find for "topk".

        Returns:
            A dictionary with result_id, category_name, filtering_method, algorithm,
            pattern_count, and top 20 patterns by support in the summary.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is not loaded."}

        df = session.dataframe
        if "category_name" not in df.columns or "text" not in df.columns:
            return {"error": "session.dataframe missing required 'category_name' or 'text' columns."}

        # 1. Filter DataFrame to specified category and non-empty text
        cat_df = df[df["category_name"] == category_name].copy()
        cat_df = cat_df[cat_df["text"].notna() & (cat_df["text"].str.strip() != "")]

        if cat_df.empty:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": f"No non-empty documents found for category '{category_name}'."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 2. Build fresh category-local DTM with English stopword removal
        vectorizer = CountVectorizer(stop_words="english")
        dtm = vectorizer.fit_transform(cat_df["text"])
        feature_names = np.array(vectorizer.get_feature_names_out())

        if dtm.shape[1] == 0:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 3. Apply feature filtering method
        dense_dtm = dtm.toarray()
        if filtering_method == "variance":
            # Keep middle 90% (5th to 95th percentile)
            variances = np.var(dense_dtm, axis=0)
            low = np.percentile(variances, 5)
            high = np.percentile(variances, 95)
            survived_mask = (variances > low) & (variances < high)
        elif filtering_method == "term_frequency":
            # Keep middle 90% (5th to 95th percentile)
            tf_sums = np.sum(dense_dtm, axis=0)
            low = np.percentile(tf_sums, 5)
            high = np.percentile(tf_sums, 95)
            survived_mask = (tf_sums > low) & (tf_sums < high)
        elif filtering_method == "tfidf":
            # Keep top 80% (>= 20th percentile)
            tfidf_trans = TfidfTransformer()
            tfidf_mat = tfidf_trans.fit_transform(dtm)
            mean_tfidf = np.array(tfidf_mat.mean(axis=0)).ravel()
            cutoff = np.percentile(mean_tfidf, 20)
            survived_mask = mean_tfidf >= cutoff
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'."}

        survived_features = feature_names[survived_mask]
        survived_dtm = dense_dtm[:, survived_mask]

        if len(survived_features) == 0:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 4. Write transactional database directly with explicit UTF-8 encoding
        # One line per document, tab-separated terms present with count >= 1
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "transactions.db")
            with open(db_path, "w", encoding="utf-8") as f:
                for row in survived_dtm:
                    present_terms = [survived_features[col_idx] for col_idx, val in enumerate(row) if val > 0]
                    if present_terms:
                        f.write("\t".join(present_terms) + "\n")

            # 5. Execute PAMI algorithm on transactional database file using mine()
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(db_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            # Use mine() method
            if hasattr(miner, "mine"):
                miner.mine()
            else:
                miner.startMine()

            # Retrieve patterns as pandas DataFrame
            patterns_df = miner.getPatternsAsDataFrame()

        patterns_list = []
        if patterns_df is not None and not patterns_df.empty:
            cols = list(patterns_df.columns)
            pattern_col = cols[0]
            support_col = cols[1] if len(cols) > 1 else cols[0]

            for _, row in patterns_df.iterrows():
                p_str = str(row[pattern_col])
                sup_val = int(row[support_col])
                patterns_list.append({"pattern": p_str, "support": sup_val})

        # Sort patterns by support descending
        patterns_list.sort(key=lambda x: x["support"], reverse=True)

        res_id = session.next_result_id("mine_patterns")
        top_20_patterns = patterns_list[:20]

        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns_list),
            "top_patterns": top_20_patterns,
        }

        # Store full pattern list in artifacts and full_report
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        df_report = pd.DataFrame(patterns_list) if patterns_list else pd.DataFrame(columns=["pattern", "support"])

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary, full_report=df_report)
        return summary

    return [mine_patterns_tool]
