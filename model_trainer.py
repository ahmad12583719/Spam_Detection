"""
=============================================================================
  model_trainer.py  
  AI-Driven Criminal Scam Analysis & Case Tracking System
=============================================================================
  PURPOSE  : Train a Naive Bayes classifier to detect scam / spam messages.
             This module is run ONCE (offline) to produce 'spam_model.pkl',
             which the live Streamlit app then loads for real-time analysis.

  AUTHOR   : Ahmad Raza
  MODULE   : AI-Powered Criminology Tools
=============================================================================
"""

# ── Standard-library imports ──────────────────────────────────────────────
import os          # For path checks (avoid re-training if model already exists)
import re          # Regular expressions for text sanitisation
import string      # Punctuation constants used in the cleaning pipeline

# ── Third-party: Data & ML ────────────────────────────────────────────────
import pandas as pd                               # Read & manipulate the CSV dataset
import joblib                                     # Serialize (pickle) trained model to disk
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text → numeric features
from sklearn.naive_bayes import MultinomialNB     # Probabilistic classifier (great for text)
from sklearn.pipeline import Pipeline             # Chain vectorizer + classifier cleanly
from sklearn.model_selection import train_test_split  # Hold-out evaluation split
from sklearn.metrics import classification_report, accuracy_score  # Evaluation metrics

# ─────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — centralise all magic strings & paths here
# ─────────────────────────────────────────────────────────────────────────
DATASET_PATH   = "spam.csv"    # Input CSV with 'label' and 'message' columns
MODEL_SAVE_PATH = "spam_model.pkl" # Output: serialised sklearn Pipeline
TEST_SIZE       = 0.25             # 25 % of data reserved for evaluation
RANDOM_SEED     = 42               # Reproducibility — same split every run
# TF-IDF hyper-parameters (chosen to balance recall on short SMS texts)
TFIDF_MAX_FEATURES = 6000         # Vocabulary cap — keeps the model lightweight
TFIDF_NGRAM_RANGE  = (1, 2)        # Unigrams + bigrams capture phrases like "click here"


# ─────────────────────────────────────────────────────────────────────────
#  STEP 1 — TEXT CLEANING FUNCTION
#  WHY: Raw SMS messages contain noise (punctuation, URLs, phone numbers,
#       currency symbols) that doesn't help classification and inflates the
#       feature space.  Cleaning improves both accuracy AND generalisation.
# ─────────────────────────────────────────────────────────────────────────
def clean_crime_evidence(raw_message: str) -> str:
    """
    Sanitise a raw SMS message for ML ingestion.

    Pipeline:
        1. Lowercase  →  removes case sensitivity
        2. Strip URLs  →  'http://bit.ly/abc' → '' (URLs are generic scam signals
                          already captured by other features)
        3. Strip digits  →  phone/account numbers add noise, not signal
        4. Remove punctuation  →  punctuation rarely aids NB classification
        5. Collapse whitespace  →  tidy up after all removals
    """
    # 1. Lowercase so 'WINNER' and 'winner' are treated as the same token
    cleaned_text = raw_message.lower()

    # 2. Remove URLs (http / https / www patterns)
    cleaned_text = re.sub(r"http\S+|www\.\S+", " ", cleaned_text)

    # 3. Remove standalone digit sequences (phone numbers, codes)
    cleaned_text = re.sub(r"\b\d+\b", " ", cleaned_text)

    # 4. Remove all punctuation characters
    cleaned_text = cleaned_text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # 5. Collapse multiple spaces into a single space and strip edge whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


# ─────────────────────────────────────────────────────────────────────────
#  STEP 2 — LOAD & VALIDATE DATASET
#  WHY: The model is only as good as its training data.  We perform basic
#       integrity checks before doing anything expensive.
# ─────────────────────────────────────────────────────────────────────────
def load_and_validate_dataset(dataset_path: str) -> pd.DataFrame:

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"[TRAINER ERROR] Dataset not found at '{dataset_path}'."
        )

    try:
        raw_dataset = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        # Kaggle's file is sometimes latin-1 encoded
        raw_dataset = pd.read_csv(dataset_path, encoding="latin-1")
    except Exception as csv_error:
        raise ValueError(f"[TRAINER ERROR] Could not parse CSV: {csv_error}")

    # Normalise column names
    raw_dataset.columns = [col.strip().lower() for col in raw_dataset.columns]

    # ── NEW: Remap Kaggle column names to your standard names ──
    kaggle_column_map = {
        "v1": "label",
        "v2": "message",
        "category": "label",   # some versions use this
        "text": "message",     # some versions use this
    }
    raw_dataset.rename(columns=kaggle_column_map, inplace=True)
    # ──────────────────────────────────────────────────────────

    required_columns = {"label", "message"}
    missing_columns  = required_columns - set(raw_dataset.columns)
    if missing_columns:
        raise ValueError(
            f"[TRAINER ERROR] Dataset missing required columns: {missing_columns}\n"
            f"  ➜  Found columns: {list(raw_dataset.columns)}"
        )

    initial_row_count  = len(raw_dataset)
    raw_dataset        = raw_dataset.dropna(subset=["label", "message"])
    dropped_row_count  = initial_row_count - len(raw_dataset)

    # ── NEW: Drop any extra Kaggle columns (v3, v4, v5 are unnamed garbage cols) ──
    raw_dataset = raw_dataset[["label", "message"]]
    # ─────────────────────────────────────────────────────────────────────────────

    print(f"  ✓ Loaded {initial_row_count} records  |  Dropped {dropped_row_count} nulls")
    print(f"  ✓ Class distribution:\n{raw_dataset['label'].value_counts().to_string()}")

    return raw_dataset

# ─────────────────────────────────────────────────────────────────────────
#  STEP 3 — FEATURE ENGINEERING
#  WHY: Machines cannot process raw strings.  TF-IDF converts each message
#       into a sparse numeric vector where each dimension represents a word/
#       bigram, weighted by how informative it is across the corpus.
#       MultinomialNB expects non-negative feature values — TF-IDF satisfies this.
# ─────────────────────────────────────────────────────────────────────────
def build_forensic_pipeline() -> Pipeline:
    """
    Construct a sklearn Pipeline that chains:
        TfidfVectorizer  →  MultinomialNB
    Using a Pipeline prevents data leakage (vectoriser is fit only on training
    data, not the test set) and makes deployment a single .predict() call.
    """
    # TF-IDF Vectorizer configuration
    #   sublinear_tf=True  : apply log(1+tf) to dampen frequency dominance
    #   min_df=1           : include tokens appearing ≥ 1 time (small dataset)
    tfidf_vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = TFIDF_NGRAM_RANGE,
        sublinear_tf = True,
        min_df       = 2,
        analyzer     = "word",
    )

    # MultinomialNB with alpha smoothing
    #   alpha=0.1 : Laplace smoothing prevents zero-probability for unseen words
    naive_bayes_classifier = MultinomialNB(alpha=0.1)

    # Package both steps into a Pipeline
    forensic_ml_pipeline = Pipeline([
        ("tfidf_vectorizer",      tfidf_vectorizer),
        ("naive_bayes_classifier", naive_bayes_classifier),
    ])

    return forensic_ml_pipeline


# ─────────────────────────────────────────────────────────────────────────
#  STEP 4 — TRAIN, EVALUATE & SAVE
# ─────────────────────────────────────────────────────────────────────────
def train_and_persist_model() -> None:
    """
    Orchestrate the full training workflow:
        Load → Clean → Split → Train → Evaluate → Save
    """
    print("\n" + "=" * 70)
    print("  AR Forensics & CyberSecurity Labs — Model Training Session")
    print("=" * 70)

    # ── 2. Load dataset ──────────────────────────────────────────────────
    print("\n[1/5] Loading & validating dataset …")
    crime_dataset = load_and_validate_dataset(DATASET_PATH)

    # ── 3. Apply text cleaning to every message ───────────────────────────
    print("\n[2/5] Cleaning evidence text …")
    crime_dataset["cleaned_message"] = crime_dataset["message"].apply(
        clean_crime_evidence
    )

    # Convert labels: 'spam' → 1 (scam/positive class), 'ham' → 0 (legitimate)
    #   WHY binary integers: sklearn metrics expect numeric class labels
    crime_dataset["binary_label"] = crime_dataset["label"].map(
        {"spam": 1, "ham": 0}
    )

    # Separate features (X) and target (y)
    crime_evidence_features = crime_dataset["cleaned_message"]
    crime_verdict_labels    = crime_dataset["binary_label"]

    # ── 4. Train / test split ─────────────────────────────────────────────
    print("\n[3/5] Splitting into training and evaluation sets …")
    (
        X_train_evidence, X_test_evidence,
        y_train_verdict,  y_test_verdict,
    ) = train_test_split(
        crime_evidence_features,
        crime_verdict_labels,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = crime_verdict_labels,   # Preserve class ratio in both splits
    )
    print(f"  ✓ Training samples : {len(X_train_evidence)}")
    print(f"  ✓ Test samples     : {len(X_test_evidence)}")

    # ── 5. Build & fit the pipeline ───────────────────────────────────────
    print("\n[4/5] Training Forensic AI Pipeline …")
    forensic_ml_pipeline = build_forensic_pipeline()
    forensic_ml_pipeline.fit(X_train_evidence, y_train_verdict)

    # ── 6. Evaluate on held-out test set ─────────────────────────────────
    print("\n[5/5] Evaluating model performance …")
    predicted_verdicts = forensic_ml_pipeline.predict(X_test_evidence)
    accuracy_score_value = accuracy_score(y_test_verdict, predicted_verdicts)

    print(f"\n  {'─'*40}")
    print(f"  ACCURACY  : {accuracy_score_value * 100:.2f}%")
    print(f"  {'─'*40}")
    print(
        classification_report(
            y_test_verdict,
            predicted_verdicts,
            target_names=["Legitimate (ham)", "Scam (spam)"],
        )
    )

    # ── 7. Persist (save) the trained pipeline to disk ───────────────────
    #   WHY joblib over pickle: joblib is optimised for numpy arrays inside
    #   sklearn objects and produces smaller, faster-loading files.
    joblib.dump(forensic_ml_pipeline, MODEL_SAVE_PATH)
    print(f"\n  ✓ Model saved  →  '{MODEL_SAVE_PATH}'")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_persist_model()
