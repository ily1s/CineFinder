from src.classification_search.smart_search_engine import (
    SmartSearchEngine,
)  # <-- import your class

INDEX_FOLDER = "data/index_data"
JSON_FOLDER = "data/Docs"


def load_engine():
    """
    Load the SmartSearchEngine using an already saved index.
    Only the JSON folder is needed to rebuild the DataFrame.
    The heavy index is loaded directly from disk.
    """
    print("Loading engine with pre-built index...")
    engine = SmartSearchEngine(
        json_folder=JSON_FOLDER,  # to load the dataframe
        load_from_file=INDEX_FOLDER,  # to load the BM25 index
    )
    return engine


def run_search(query, top_n=10):
    engine = load_engine()
    results = engine.search(query, top_n=top_n)
    return results


if __name__ == "__main__":
    query = "action movies of christopher nolan 2010"
    results = run_search(query, top_n=10)

    print("\n=== Search Results ===")
    print(results[["Title", "Director", "Genres", "Release_Date", "score"]])
