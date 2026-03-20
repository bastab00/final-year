{"id":"58391","variant":"standard","title":"text_cleaner.py — Preprocessing Pipeline"}
import json
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Full preprocessing pipeline for inflation abstracts."""
        
        # Lowercase
        text = text.lower()

        # Remove digits
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = text.split()

        # Remove stopwords & lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return " ".join(tokens)

    def preprocess_dataset(self, input_path: str, output_path: str):
        """Loads JSON, cleans text, and writes cleaned CSV."""
        
        # Read JSON file (list of dicts)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Convert label from string to int
        df["Label"] = df["Label"].astype(int)

        # Apply cleaning
        df["Cleaned_Abstract"] = df["Abstract"].apply(self.clean_text)

        # Save processed dataset
        df.to_csv(output_path, index=False)
        print(f"[INFO] Cleaned dataset saved to: {output_path}")


# ------------------------------------------------------------------------------------------
# Run this file directly to preprocess your dataset
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.preprocess_dataset(
        input_path="../../data/raw/inflation_dataset.json",
        output_path="../../data/processed/cleaned_inflation_dataset.csv"

    )
