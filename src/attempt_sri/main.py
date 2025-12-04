# main.py
from search_engine import MovieSearchEngine

def main():
    print("\n" + "="*80)
    print("      CINEFINDER PRO - THE ULTIMATE MOVIE SEARCH ENGINE")
    print("        Lemmatization + WordNet Synonyms + BM25 + Your Perfect Index")
    print("="*80)

    DOCS_DIR = r'D:\GI5\BOUZID\Projet\CineFinder\data\docs'
    engine = MovieSearchEngine()
    engine.index_documents(DOCS_DIR)

    print("\n" + "="*80)
    print("READY! Type any query (e.g., 'space war', 'funny robot', 'dark knight')")
    print("="*80)

    debug = False
    while True:
        q = input("\nSearch: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using CineFinder PRO!")
            break
        if q.lower() == 'debug':
            debug = not debug
            print(f"Debug: {'ON' if debug else 'OFF'}")
            continue
        if q.lower() == 'stats':
            engine.index.print_statistics()
            continue
        if not q:
            continue

        results = engine.search(q, top_k=10, debug=debug)
        engine.display_results(results)

if __name__ == "__main__":
    main()