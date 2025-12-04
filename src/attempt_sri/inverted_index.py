from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set



class InvertedIndex:
    """
    In-memory inverted index - rebuilt each time program runs
    """

    def __init__(self):
        # Core data structures
        self.index = defaultdict(list)  # term → [(doc_id, frequency), ...]
        self.doc_lengths = {}  # doc_id → number of terms
        self.doc_metadata = {}  # doc_id → full document data
        self.vocabulary = set()  # all unique terms

        # Statistics
        self.total_docs = 0
        self.avg_doc_length = 0

        print("[Index] Inverted index initialized (in-memory)")

    def add_document(self, doc_id: str, terms: List[str], metadata: dict, debug=False):
        """
        Add document to index with detailed tracing
        """
        if debug:
            print(f"\n[Index] Adding document: {doc_id}")
            print(f"  Terms in document: {len(terms)}")

        # Count term frequencies
        term_freq = Counter(terms)

        if debug:
            print(f"  Unique terms: {len(term_freq)}")
            print(f"  Top terms: {term_freq.most_common(5)}")

        # Store metadata
        self.doc_metadata[doc_id] = metadata
        self.doc_lengths[doc_id] = len(terms)

        # Update inverted index
        for term, freq in term_freq.items():
            self.index[term].append((doc_id, freq))
            self.vocabulary.add(term)

        self.total_docs += 1

        if debug:
            print(f"  ✓ Document indexed successfully")

    def finalize(self):
        """
        Calculate statistics after all documents added
        """
        print("\n[Index] Finalizing index...")

        # Calculate average document length
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

        # Sort posting lists by document ID (for efficiency)
        for term in self.index:
            self.index[term].sort(key=lambda x: x[0])

        print(f"[Index] ✓ Finalized")
        self.print_statistics()

    def get_posting_list(self, term: str) -> List[Tuple[str, int]]:
        """Get posting list for a term"""
        return self.index.get(term, [])

    def get_document_frequency(self, term: str) -> int:
        """Number of documents containing term"""
        return len(self.index.get(term, []))

    def print_statistics(self):
        """Display index statistics"""
        print("\n" + "=" * 70)
        print("INVERTED INDEX STATISTICS")
        print("=" * 70)
        print(f"Total documents: {self.total_docs}")
        print(f"Vocabulary size: {len(self.vocabulary)} unique terms")
        print(f"Average document length: {self.avg_doc_length:.2f} terms")
        print(f"Total postings: {sum(len(postings) for postings in self.index.values())}")

        # Show sample entries
        print(f"\n[Sample Index Entries]")
        sample_terms = sorted(list(self.vocabulary))[:5]
        for term in sample_terms:
            postings = self.index[term]
            print(f"  '{term}' → appears in {len(postings)} documents")
            print(f"    Example postings: {postings[:3]}")
        print("=" * 70)
