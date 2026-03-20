import pandas as pd
import spacy
from gensim import corpora, models
import gensim

# ===============================
# Load dataset
# ===============================
print("[INFO] Loading data...")
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

texts = df["Cleaned_Abstract"].astype(str).tolist()

# ===============================
# spaCy tokenizer (fast + stable)
# ===============================
print("[INFO] Loading spaCy tokenizer...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def tokenize(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]

print("[INFO] Tokenizing texts...")
tokenized_texts = [tokenize(t) for t in texts]

# ===============================
# Create dictionary + corpus
# ===============================
print("[INFO] Creating dictionary and corpus...")

dictionary = corpora.Dictionary(tokenized_texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)

corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# ===============================
# LDA Model (Optimized for low hardware)
# ===============================
print("[INFO] Training LDA model...")

lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    random_state=42,
    passes=6,           # light-medium training
    chunksize=150,
    alpha='auto',
    per_word_topics=False
)

print("\n===== TOPICS DISCOVERED (LDA) =====\n")
topics = lda_model.print_topics(num_words=10)
for idx, topic in topics:
    print(f"Topic {idx}: {topic}")

# ===============================
# Save model
# ===============================
lda_model.save("lda_model.model")
dictionary.save("lda_dictionary.dict")

print("\n[INFO] LDA model saved successfully!")
