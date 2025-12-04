from typing import List, Set
import nltk
import ssl
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Fix SSL + Auto-download required data (UPDATED: Includes English-specific tagger)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

REQUIRED = [
    'punkt', 'punkt_tab',           # Tokenization
    'stopwords', 'wordnet', 'omw-1.4',  # Core NLP
    'averaged_perceptron_tagger',   # Generic tagger (fallback)
    'averaged_perceptron_tagger_eng'  # ← NEW: English-specific POS tagger (fixes your error!)
]

print("Checking required NLTK packages...")
for pkg in REQUIRED:
    try:
        if pkg in ['punkt', 'punkt_tab']:
            nltk.data.find(f'tokenizers/{pkg}')
        elif 'perceptron' in pkg:
            # For taggers: generic is 'taggers/averaged_perceptron_tagger', eng is 'taggers/averaged_perceptron_tagger_eng'
            base_path = 'taggers/averaged_perceptron_tagger'
            full_path = f'{base_path}_eng' if '_eng' in pkg else base_path
            nltk.data.find(full_path)
        else:
            nltk.data.find(f'corpora/{pkg}')
        print(f"   ✓ {pkg} ready")
    except LookupError:
        print(f"   Downloading {pkg}...")
        nltk.download(pkg, quiet=False)
print("All NLTK data ready!\n")


class TextPreprocessor:
    def __init__(self):
        self.stop_words: Set[str] = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Keep negation words (important!)
        for w in ['not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', 'couldnt', 'wouldnt']:
            self.stop_words.discard(w.lower())

        print(f"TextPreprocessor ready → Lemmatization + POS + WordNet Synonyms")

    def _pos_tag_to_wordnet(self, tag: str):
        if tag.startswith('J'): return wordnet.ADJ
        if tag.startswith('V'): return wordnet.VERB
        if tag.startswith('N'): return wordnet.NOUN
        if tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    def _get_synonyms(self, word: str) -> Set[str]:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                s = lemma.name().replace('_', ' ').lower()
                if s != word.lower():
                    synonyms.add(s)
        return synonyms

    def process(self, text: str, expand_synonyms: bool = False) -> List[str]:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
        tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatize with POS
        tagged = pos_tag(tokens)
        tokens = [self.lemmatizer.lemmatize(w, self._pos_tag_to_wordnet(t)) for w, t in tagged]

        if expand_synonyms:
            expanded = []
            for t in tokens:
                expanded.append(t)
                expanded.extend(self._get_synonyms(t))
            tokens = list(dict.fromkeys(expanded))  # preserve order
        else:
            tokens = list(dict.fromkeys(tokens))

        return tokens