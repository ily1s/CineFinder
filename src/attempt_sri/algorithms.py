import math
from typing import Dict, List, Tuple, Set
from inverted_index import InvertedIndex

class TFIDFRanker:
    """TF-IDF Ranking with detailed explanations"""

    def __init__(self, index: InvertedIndex):
        self.index = index
        # Precompute IDF values for efficiency
        self.idf_cache = {}
        print("\n[TF-IDF] Precomputing IDF values...")
        for term in index.vocabulary:
            self.idf_cache[term] = self._compute_idf(term)
        print(f"[TF-IDF] ✓ Computed IDF for {len(self.idf_cache)} terms")

    def _compute_idf(self, term: str) -> float:
        """
        IDF = log(N / df)
        N = total documents
        df = document frequency (documents containing term)
        """
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(self.index.total_docs / df)

    def _compute_tf(self, term_freq: int, doc_length: int) -> float:
        """
        TF = term_freq / doc_length
        Normalized by document length
        """
        if doc_length == 0:
            return 0.0
        return term_freq / doc_length

    def score(self, query_terms: List[str], doc_id: str, debug=False) -> float:
        """Calculate TF-IDF score"""
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 1)

        if debug:
            print(f"\n  [TF-IDF Scoring] Document: {doc_id}")
            print(f"    Document length: {doc_length} terms")

        for term in query_terms:
            # Find term frequency in this document
            tf_in_doc = 0
            for doc, freq in self.index.get_posting_list(term):
                if doc == doc_id:
                    tf_in_doc = freq
                    break

            if tf_in_doc > 0:
                tf = self._compute_tf(tf_in_doc, doc_length)
                idf = self.idf_cache.get(term, 0.0)
                term_score = tf * idf
                score += term_score

                if debug:
                    print(f"    Term '{term}': TF={tf:.4f}, IDF={idf:.4f}, Score={term_score:.4f}")

        if debug:
            print(f"    → Total Score: {score:.4f}")

        return score


class BM25Ranker:
    """BM25 Ranking with detailed explanations"""

    def __init__(self, index: InvertedIndex, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1  # Term frequency saturation
        self.b = b  # Document length normalization

        # Precompute IDF values
        self.idf_cache = {}
        print(f"\n[BM25] Initializing with k1={k1}, b={b}")
        print("[BM25] Precomputing IDF values...")
        for term in index.vocabulary:
            self.idf_cache[term] = self._compute_idf(term)
        print(f"[BM25] ✓ Computed IDF for {len(self.idf_cache)} terms")

    def _compute_idf(self, term: str) -> float:
        """
        BM25 IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        More refined than standard IDF
        """
        N = self.index.total_docs
        df = self.index.get_document_frequency(term)
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_terms: List[str], doc_id: str, debug=False) -> float:
        """
        Calculate BM25 score
        Formula: IDF(term) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × doc_len/avg_len))
        """
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 1)
        avgdl = self.index.avg_doc_length

        if debug:
            print(f"\n  [BM25 Scoring] Document: {doc_id}")
            print(f"    Doc length: {doc_length}, Avg length: {avgdl:.2f}")
            print(f"    Parameters: k1={self.k1}, b={self.b}")

        for term in query_terms:
            # Find term frequency in document
            tf = 0
            for doc, freq in self.index.get_posting_list(term):
                if doc == doc_id:
                    tf = freq
                    break

            if tf > 0:
                idf = self.idf_cache.get(term, 0.0)

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl))
                term_score = idf * (numerator / denominator)
                score += term_score

                if debug:
                    print(f"    Term '{term}':")
                    print(f"      TF={tf}, IDF={idf:.4f}")
                    print(f"      Numerator={numerator:.4f}, Denominator={denominator:.4f}")
                    print(f"      Score={term_score:.4f}")

        if debug:
            print(f"    → Total Score: {score:.4f}")

        return score
