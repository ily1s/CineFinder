# search_engine.py
import json
import os
from typing import List, Tuple
from text_processing import TextPreprocessor
from inverted_index import InvertedIndex
from algorithms import BM25Ranker


class MovieSearchEngine:
    def __init__(self):
        print("\n" + "="*80)
        print("      CINEFINDER PRO - MAXIMUM SEARCH QUALITY EDITION")
        print("="*80)

        self.preprocessor = TextPreprocessor()
        self.index = InvertedIndex()                    # ← Your excellent index
        self.ranker = None  # ← Don't create BM25 yet!

        print("Ranking → BM25 (Industry Standard)")
        print("Preprocessing → Lemmatization + POS + WordNet Synonyms")
        print("Synonym Expansion → Enabled on queries")

    def index_documents(self, docs_dir: str, debug: bool = False):
        print("\n" + "="*80)
        print("INDEXING MOVIE COLLECTION")
        print("="*80)

        json_files = [f for f in os.listdir(docs_dir) if f.endswith('.json')]
        total = len(json_files)
        print(f"Found {total} movies → starting indexing...\n")

        for i, filename in enumerate(json_files, 1):
            path = os.path.join(docs_dir, filename)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = ' '.join([
                str(data.get('Title', '')),
                str(data.get('Overview', '')),
                str(data.get('Tagline', '')),
                str(data.get('Genres', '')),
                str(data.get('Director', '')),
                str(data.get('Cast', ''))
            ])

            terms = self.preprocessor.process(text, expand_synonyms=False)
            doc_id = filename.replace('.json', '')
            self.index.add_document(doc_id, terms, data, debug=(debug and i == 1))

            if i % 20 == 0 or i == total:
                print(f"   Indexed {i}/{total} movies...")

        self.index.finalize()
        # CRITICAL FIX: Initialize BM25 AFTER finalize()!
        from algorithms import BM25Ranker
        self.ranker = BM25Ranker(self.index)  # ← NOW it sees all terms!

        print(f"\nINDEX READY → {self.index.total_docs} movies indexed")
        print(f"Vocabulary size: {len(self.index.vocabulary)} unique terms")
        print("BM25 ranker initialized with real IDF values!")

    def search(self, query: str, top_k: int = 10, debug: bool = False) -> List[Tuple[str, float, dict]]:
        print("\n" + "-"*80)
        print(f"SEARCH → '{query}'")
        print("-"*80)

        query_terms = self.preprocessor.process(query, expand_synonyms=True)
        if debug:
            print(f"Expanded query terms: {query_terms}")

        if not query_terms:
            return []

        candidates = set()
        for term in query_terms:
            for doc_id, _ in self.index.get_posting_list(term):
                candidates.add(doc_id)

        print(f"Found {len(candidates)} candidate movies")

        results = []
        for doc_id in candidates:
            score = self.ranker.score(query_terms, doc_id, debug=debug)
            if score > 0:
                results.append((doc_id, score, self.index.doc_metadata[doc_id]))

        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Top {min(top_k, len(results))} results ready!\n")
        return results[:top_k]

    def display_results(self, results: List[Tuple[str, float, dict]]):
        print("\n" + "="*80)
        print("TOP RESULTS")
        print("="*80)
        if not results:
            print("No movies found")
            return
        for rank, (doc_id, score, meta) in enumerate(results, 1):
            print(f"\n#{rank} | Score: {score:.4f}")
            print(f"Title: {meta.get('Title', 'N/A')}")
            print(f"Director: {meta.get('Director', 'N/A')} | Year: {meta.get('Release_Date', '')[:4]}")
            print(f"Genres: {meta.get('Genres', 'N/A')}")
            print(f"Rating: {meta.get('Vote_Average', 'N/A')}/10")
            overview = meta.get('Overview', '')[:300]
            if len(meta.get('Overview', '')) > 300:
                overview += "..."
            print(f"Overview: {overview}")
        print("\n" + "="*80)