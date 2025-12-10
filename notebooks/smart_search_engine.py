import pandas as pd
import numpy as np
from collections import defaultdict
import re
import math
import spacy
import json
import os
from pathlib import Path

class SmartSearchEngine:
    def __init__(self, df=None, json_folder=None, load_from_file=None):
        """
        Initialize the search engine with either:
        - df: A pandas DataFrame (original behavior)
        - json_folder: Path to folder containing JSON files
        - load_from_file: Path to pre-built index
        """
        # Load data from JSON folder if provided
        if json_folder is not None:
            print(f"Loading JSON files from: {json_folder}")
            self.df = self.load_json_files(json_folder)
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Must provide either df or json_folder")
        
        self.N = len(self.df)
        
        # Load spaCy model
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define fields to index
        self.fields = ['Title', 'Director', 'Genres', 'Overview', 'Release_Date']
        
        if load_from_file:
            # Load index from file
            self.load_index(load_from_file)
        else:
            # Build index from scratch
            self.inverted_index = defaultdict(lambda: defaultdict(list))
            self.doc_lengths = {}
            self.avg_doc_length = {}
            self.directors_set = set()
            self.genres_set = set()
            self.title_words = set()
            self.years_set = set()
            
            self.build_index()
            self.build_recognition_dicts()
    
    @staticmethod
    def load_json_files(folder_path):
        """
        Load all JSON files from a folder and combine into a DataFrame.
        Each JSON file should contain one movie record.
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all JSON files
        json_files = list(folder.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in: {folder_path}")
        
        print(f"Found {len(json_files)} JSON files")
        
        # Load all JSON files
        movies = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    movie_data = json.load(f)
                    movies.append(movie_data)
            except Exception as e:
                print(f"Warning: Could not load {json_file.name}: {e}")
        
        print(f"Successfully loaded {len(movies)} movies")
        
        # Convert to DataFrame
        df = pd.DataFrame(movies)
        
        # Ensure required columns exist
        required_columns = ['Title', 'Director', 'Genres', 'Overview', 'Release_Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            for col in missing_columns:
                df[col] = None
        
        return df
    
    def preprocess_text(self, text, is_date=False):
        """Clean and tokenize with spaCy + lemmatization"""
        if pd.isna(text):
            return []
        text = str(text).lower()
        
        # For dates, extract year (format: YYYY-MM-DD or just YYYY)
        if is_date:
            year_match = re.findall(r'\b(?:19|20)\d{2}\b', text)
            return year_match if year_match else []
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract lemmas with spaCy's built-in stopwords
        tokens = [
            token.lemma_ 
            for token in doc 
            if not token.is_stop          # spaCy's built-in stopwords
            and not token.is_punct        # Remove punctuation
            and not token.is_space        # Remove whitespace
            and len(token.lemma_) > 2     # Remove short tokens
            and token.is_alpha            # Keep only alphabetic tokens
        ]
        
        return tokens
    
    def build_index(self):
        """Build inverted index by field"""
        print("Building inverted index...")
        
        for field in self.fields:
            self.doc_lengths[field] = {}
            self.avg_doc_length[field] = 0
        
        for idx, row in self.df.iterrows():
            for field in self.fields:
                # Special treatment for dates
                is_date_field = (field == 'Release_Date')
                tokens = self.preprocess_text(row[field], is_date=is_date_field)
                self.doc_lengths[field][idx] = len(tokens)
                
                term_freq = defaultdict(int)
                for token in tokens:
                    term_freq[token] += 1
                
                for term, freq in term_freq.items():
                    self.inverted_index[field][term].append((idx, freq))
        
        for field in self.fields:
            if self.doc_lengths[field]:
                self.avg_doc_length[field] = np.mean(list(self.doc_lengths[field].values()))
        
        print(f"Index built: {len(self.df)} documents indexed")
    
    def build_recognition_dicts(self):
        """Build dictionaries to automatically recognize terms"""
        print("Building recognition dictionaries...")
        
        # Extract all directors
        for director in self.df['Director'].dropna().unique():
            tokens = self.preprocess_text(director)
            self.directors_set.update(tokens)
        
        # Extract all genres
        for genres in self.df['Genres'].dropna():
            for genre in str(genres).split(','):
                tokens = self.preprocess_text(genre.strip())
                self.genres_set.update(tokens)
        
        # Extract important words from titles
        for title in self.df['Title'].dropna():
            tokens = self.preprocess_text(title)
            self.title_words.update(tokens)
        
        # Extract all years from release dates
        for date in self.df['Release_Date'].dropna():
            years = self.preprocess_text(date, is_date=True)
            self.years_set.update(years)
        
        print(f"Unique directors: {len(self.directors_set)}")
        print(f"Unique genres: {len(self.genres_set)}")
        print(f"Available years: {len(self.years_set)}")
    
    def classify_query_terms(self, query_tokens):
        """Automatically classify each query term"""
        classified = {
            'director': [],
            'genre': [],
            'title': [],
            'year': [],
            'general': []
        }
        
        for term in query_tokens:
            # Check if it's a year (4 digits starting with 19 or 20)
            if re.match(r'^(?:19|20)\d{2}$', term):
                classified['year'].append(term)
            # Check in which field the term appears most
            elif term in self.directors_set:
                classified['director'].append(term)
            elif term in self.genres_set:
                classified['genre'].append(term)
            elif term in self.title_words:
                classified['title'].append(term)
            else:
                # General term, search everywhere
                classified['general'].append(term)
        
        return classified
    
    def bm25_score(self, term, doc_id, field, k1=1.5, b=0.75):
        """Calculate BM25 score for a term in a document"""
        if term not in self.inverted_index[field]:
            return 0.0
        
        tf = 0
        for doc, freq in self.inverted_index[field][term]:
            if doc == doc_id:
                tf = freq
                break
        
        if tf == 0:
            return 0.0
        
        df = len(self.inverted_index[field][term])
        idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        
        doc_len = self.doc_lengths[field].get(doc_id, 0)
        avg_len = self.avg_doc_length[field]
        
        if avg_len == 0:
            return 0.0
        
        norm = 1 - b + b * (doc_len / avg_len)
        score = idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        
        return score
    
    def search(self, query, top_n=10):
        """Smart search with automatic term classification"""
        # Extract years BEFORE preprocessing
        years_in_query = re.findall(r'\b(?:19|20)\d{2}\b', query)
        
        # Tokenize query with spaCy + lemmatization
        query_tokens = self.preprocess_text(query)
        
        # Add extracted years to tokens
        query_tokens.extend(years_in_query)
        
        if not query_tokens:
            return pd.DataFrame()
        
        # Classify query terms
        classified = self.classify_query_terms(query_tokens)
        
        # Debug: display classification
        print(f"\n=== Term Classification ===")
        for category, terms in classified.items():
            if terms:
                print(f"{category.capitalize()}: {terms}")
        
        # Collect all candidate documents
        candidate_docs = set()
        
        # Search terms in their corresponding fields
        field_mapping = {
            'director': ['Director'],
            'genre': ['Genres'],
            'title': ['Title'],
            'year': ['Release_Date'],
            'general': ['Title', 'Director', 'Genres', 'Overview']
        }
        
        for category, terms in classified.items():
            target_fields = field_mapping[category]
            for term in terms:
                for field in target_fields:
                    if term in self.inverted_index[field]:
                        for doc_id, _ in self.inverted_index[field][term]:
                            candidate_docs.add(doc_id)
        
        # Calculate scores for each document
        scores = {}
        for doc_id in candidate_docs:
            total_score = 0.0
            
            # Score for director terms
            for term in classified['director']:
                score = self.bm25_score(term, doc_id, 'Director')
                total_score += score * 1.0
            
            # Score for genre terms
            for term in classified['genre']:
                score = self.bm25_score(term, doc_id, 'Genres')
                total_score += score * 1.0
            
            # Score for title terms
            for term in classified['title']:
                score = self.bm25_score(term, doc_id, 'Title')
                total_score += score * 1.0
            
            # Score for years
            for term in classified['year']:
                score = self.bm25_score(term, doc_id, 'Release_Date')
                total_score += score * 1.0
            
            # Score for general terms
            for term in classified['general']:
                for field in ['Title', 'Director', 'Genres', 'Overview', 'Release_Date']:
                    score = self.bm25_score(term, doc_id, field)
                    total_score += score * 1.0
            
            scores[doc_id] = total_score
        
        # Sort by descending score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_docs:
            return pd.DataFrame()
        
        result_indices = [doc_id for doc_id, _ in sorted_docs]
        result_scores = [score for _, score in sorted_docs]
        
        results = self.df.loc[result_indices, ['Title', 'Overview', 'Genres', 'Director', 'Release_Date']].copy()
        results['score'] = result_scores
        
        return results.reset_index(drop=True)
    
    def save_index(self, folder_path="../data/index_data"):
        """Save inverted index and metadata to JSON"""
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"\nSaving index to '{folder_path}'...")
        
        # Convert inverted_index to serializable format
        serializable_index = {}
        for field, terms_dict in self.inverted_index.items():
            serializable_index[field] = {}
            for term, postings in terms_dict.items():
                serializable_index[field][term] = [[int(doc_id), int(freq)] for doc_id, freq in postings]
        
        # Save inverted index
        index_path = os.path.join(folder_path, "inverted_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            'N': self.N,
            'doc_lengths': {field: {int(k): v for k, v in lengths.items()} 
                           for field, lengths in self.doc_lengths.items()},
            'avg_doc_length': self.avg_doc_length,
            'directors_set': list(self.directors_set),
            'genres_set': list(self.genres_set),
            'title_words': list(self.title_words),
            'years_set': list(self.years_set),
            'fields': self.fields
        }
        
        metadata_path = os.path.join(folder_path, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Display stats
        index_size = os.path.getsize(index_path) / 1024
        metadata_size = os.path.getsize(metadata_path) / 1024
        
        print(f"✓ Index saved successfully!")
        print(f"  📁 Folder: {folder_path}")
        print(f"  📄 inverted_index.json: {index_size:.2f} KB")
        print(f"  📄 metadata.json: {metadata_size:.2f} KB")
        print(f"  📊 Total: {index_size + metadata_size:.2f} KB")
    
    def load_index(self, folder_path="../data"):
        """Load inverted index from JSON"""
        print(f"\nLoading index from '{folder_path}'...")
        
        index_path = os.path.join(folder_path, "inverted_index.json")
        metadata_path = os.path.join(folder_path, "metadata.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"File not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"File not found: {metadata_path}")
        
        # Load inverted index
        with open(index_path, 'r', encoding='utf-8') as f:
            serializable_index = json.load(f)
        
        # Rebuild defaultdict structure with tuples
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        for field, terms_dict in serializable_index.items():
            for term, postings in terms_dict.items():
                self.inverted_index[field][term] = [(doc_id, freq) for doc_id, freq in postings]
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.N = metadata['N']
        self.doc_lengths = {field: {int(k): v for k, v in lengths.items()} 
                           for field, lengths in metadata['doc_lengths'].items()}
        self.avg_doc_length = metadata['avg_doc_length']
        self.directors_set = set(metadata['directors_set'])
        self.genres_set = set(metadata['genres_set'])
        self.title_words = set(metadata['title_words'])
        self.years_set = set(metadata['years_set'])
        self.fields = metadata['fields']
        
        print(f"✓ Index loaded successfully!")


