# %%
# 3.i / 3.ii (Lexicons studied/used in code: VADER + TextBlob)
import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
# 3.i (VADER)
from nltk.sentiment import SentimentIntensityAnalyzer
# 3.ii (TextBlob)
from textblob import TextBlob
# 7 (accuracy/precision/recall/F1 + confusion matrix)
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix)

# %%
# 10.
# =========================
# CONFIG (AMAZON_FASHION_5.json.gz)
# =========================

DATASET_PATH = "AMAZON_FASHION_5.json.gz"

OUTPUT_DIR = "phase1_outputs"

RANDOM_SEED = 42
SAMPLE_SIZE = 1000

# When True the notebook will both save and display figures during execution.
SHOW_PLOTS = True

# %%
# =========================
# UTILS
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# %%
# 10 (submit documented code: load and normalize required dataset)
def load_reviews_dataset(path: str) -> pd.DataFrame:
    """
    Loads the AMAZON_FASHION_5 review file and normalizes key columns so the
    rest of the Phase 1 workflow can stay close to the original notebook.
    """
    df = pd.read_json(path, lines=True, compression="gzip")
    df = df.rename(
        columns={
            "overall": "ratings",
            "reviewText": "review_text",
            "reviewerID": "reviewer_id",
        }
    )
    return df.copy()

# %%
# 2.b (choose the appropriate columns for the sentiment analyzer)
def to_text_maybe_list(x) -> str:
    """
    Some datasets store tokens like: "['good','shirt']" as a string.
    Convert that into: "good shirt".
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                return " ".join(str(t) for t in lst).strip()
        except Exception:
            pass

    return s

# %%
# 2.b (combine chosen text columns into one sentiment input)
def combine_summary_review(summary_text: str, review_text: str) -> str:
    summary_text = (summary_text or "").strip()
    review_text = (review_text or "").strip()
    if summary_text and review_text:
        return f"{summary_text}. {review_text}"
    return summary_text or review_text

# %%
# 2.a (label data based on the rating of the product)
def label_from_rating(r):
    """
    Label rules required by the Phase 1 rubric.
    """
    try:
        r = float(r)
    except Exception:
        return np.nan

    # 2.a.i. (Ratings 4,5 => Positive)
    if r >= 4:
        return "Positive"
    # 2.a.ii. (Rating 3 => Neutral)
    elif r == 3:
        return "Neutral"
    # 2.a.iii. (Ratings 1,2 => Negative)
    elif r <= 2:
        return "Negative"
    return np.nan

# %%
# 4 (VADER preprocessing based on the selected lexicon characteristics)
def preprocess_for_vader(text: str) -> str:
    # 4 (keep punctuation/emphasis, only remove URLs and normalize spaces)
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# %%
# 4 (TextBlob preprocessing based on the selected lexicon characteristics)
def preprocess_for_textblob(text: str) -> str:
    # 4 (normalize lowercase text, remove URLs, and remove most non-letters)
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# %%
# 1.a / 1.b / 1.e (plots used for dataset exploration)
def save_hist(values, title, xlabel, filename, bins=50):
    plt.figure()
    plt.hist(pd.Series(values).dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# %%
# 1.c / 1.d / 1.e (boxplots used for dataset exploration)
def save_boxplot(values, title, ylabel, filename):
    plt.figure()
    plt.boxplot(pd.Series(values).dropna(), vert=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# %%
# 7 (confusion matrix outputs for validation)
def save_confusion(cm, labels, title, filename):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# %%
# 7 (accuracy/precision/recall/F1 + confusion matrix)
def evaluate(y_true, y_pred, labels=("Negative", "Neutral", "Positive")):
    acc = accuracy_score(y_true, y_pred)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return acc, pr_w, rc_w, f1_w, cm

# %%
# =========================
# MODELS (Lexicon)
# =========================
# 6.a / 3.i (VADER lexicon model)
def vader_predict(texts: pd.Series) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()
    preds = []
    for t in texts:
        c = analyzer.polarity_scores(t)["compound"]
        if c >= 0.05:
            preds.append("Positive")
        elif c <= -0.05:
            preds.append("Negative")
        else:
            preds.append("Neutral")
    return pd.Series(preds, index=texts.index)

# %%
# 6.a / 3.ii (TextBlob lexicon model)
def textblob_predict(texts: pd.Series) -> pd.Series:
    preds = []
    for t in texts:
        pol = TextBlob(t).sentiment.polarity
        if pol > 0.1:
            preds.append("Positive")
        elif pol < -0.1:
            preds.append("Negative")
        else:
            preds.append("Neutral")
    return pd.Series(preds, index=texts.index)

# %%
# =========================
# MAIN
# =========================

# Setup output directory
# 10 (submit documented code)
ensure_dir(OUTPUT_DIR)

# %%
# -------------------------
# Load AMAZON_FASHION_5.json.gz
# -------------------------
# 10 (submit documented code)
df_train = load_reviews_dataset(DATASET_PATH)

print("Dataset shape:", df_train.shape)
print("Available columns:", sorted(df_train.columns))

# %% [markdown]
# ## Phase 1 Dataset Choice
# This notebook now uses only `AMAZON_FASHION_5.json.gz`, which is the required Phase 1 source file.
# The code normalizes the dataset schema once after loading so the rest of the analysis can stay close to the original workflow.

# %%
# -------------------------
# Dataset data exploration
# -------------------------
# 1.a (counts and averages)
df_train["ratings"] = pd.to_numeric(df_train["ratings"], errors="coerce")
n_before = len(df_train)
df_train = df_train.dropna(subset=["ratings"]).copy()
n_after = len(df_train)

avg_rating = df_train["ratings"].mean()
med_rating = df_train["ratings"].median()
n_users = df_train["reviewer_id"].nunique() if "reviewer_id" in df_train.columns else None
n_products = df_train["asin"].nunique() if "asin" in df_train.columns else None
missing_review_text = int(df_train["review_text"].isna().sum()) if "review_text" in df_train.columns else None

print("Rows before rating cleanup:", n_before)
print("Rows after rating cleanup :", n_after)
print("Average rating:", round(avg_rating, 3))
print("Median rating :", med_rating)
print("Unique users  :", n_users)
print("Unique products:", n_products)
print("Missing review_text rows:", missing_review_text)

save_hist(df_train["ratings"], "1.a Ratings Distribution", "rating", "1a_ratings_hist.png", bins=10)

# %%
# Reviews per product / user

if "asin" in df_train.columns:
    # 1.b.
    reviews_per_product = df_train["asin"].value_counts()
    print("Products with at least one review:", len(reviews_per_product))
    print("Average reviews per product:", round(reviews_per_product.mean(), 3))
    save_hist(
        reviews_per_product.values,
        "1.b Distribution of Reviews Across Products",
        "reviews per product",
        "1b_reviews_across_products_hist.png",
        bins=60
    )
    # 1.c.
    save_boxplot(
        reviews_per_product.values,
        "1.c Reviews per Product (Boxplot)",
        "reviews per product",
        "1c_reviews_per_product_boxplot.png"
    )

if "reviewer_id" in df_train.columns:
    # 1.d.
    reviews_per_user = df_train["reviewer_id"].value_counts()
    print("Users with at least one review:", len(reviews_per_user))
    print("Average reviews per user:", round(reviews_per_user.mean(), 3))
    save_hist(
        reviews_per_user.values,
        "1.d Distribution of Reviews per User",
        "reviews per user",
        "1d_reviews_per_user_hist.png",
        bins=60
    )
    save_boxplot(
        reviews_per_user.values,
        "1.d Reviews per User (Boxplot)",
        "reviews per user",
        "1d_reviews_per_user_boxplot.png"
    )

# %%
# 1.b / 1.c (identify which product ASIN has the most reviews)
if "asin" in df_train.columns:
    top_reviewed_products = df_train["asin"].value_counts().head(10)
    most_reviewed_asin = top_reviewed_products.index[0]
    most_reviewed_count = int(top_reviewed_products.iloc[0])

    plt.figure(figsize=(10, 5))
    top_reviewed_products.sort_values().plot(kind="barh")
    plt.title("Top 10 Product ASINs by Number of Reviews")
    plt.xlabel("Number of reviews")
    plt.ylabel("ASIN")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1b_top_reviewed_products.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    print("Product ASIN with the most reviews:", most_reviewed_asin)
    print("Number of reviews for that product ASIN:", most_reviewed_count)

# %%
# 1.d. / 1.g. (check whether reviews are coming from the same user)
reviewer_counts = df_train["reviewer_id"].value_counts()
users_with_multiple_reviews = int((reviewer_counts > 1).sum())
repeated_user_product_pairs = int(df_train.duplicated(subset=["reviewer_id", "asin"]).sum())

print("Number of unique users:", int(df_train["reviewer_id"].nunique()))
print("Number of users with multiple reviews:", users_with_multiple_reviews)
print("Number of repeated reviewer_id + asin pairs:", repeated_user_product_pairs)

if repeated_user_product_pairs > 0:
    print("Conclusion: the dataset includes repeated user-product review pairs, so repeat-user behavior should be noted in the Phase 1 report.")
else:
    print("Conclusion: the dataset does not show repeated user-product review pairs in this Phase 1 check.")


# %%
# 1.g histogram visualization for repeated reviewer-product behavior
df_pairs = df_train.groupby("reviewer_id").agg(
    total_reviews=("asin", "count"),
    unique_products=("asin", "nunique")
).reset_index()
df_pairs["repeated_reviews"] = df_pairs["total_reviews"] - df_pairs["unique_products"]

save_hist(
    df_pairs["repeated_reviews"].values,
    "1.g Distribution of Repeated Reviews per Reviewer",
    "repeated reviews per reviewer",
    "1g_repeated_reviews_per_reviewer_hist.png",
    bins=20
)

print("Histogram saved: 1g_repeated_reviews_per_reviewer_hist.png")

# %%
# Outlier detection (IQR)
# 2.c (check for outliers)
if "review_text_str" not in df_train.columns:
    df_train["review_text_str"] = df_train["review_text"].apply(to_text_maybe_list)

if "summary_str" not in df_train.columns:
    df_train["summary_str"] = df_train["summary"].apply(to_text_maybe_list)

if "review_len_words" not in df_train.columns or "review_len_chars" not in df_train.columns:
    df_train["review_len_words"] = df_train["review_text_str"].apply(lambda x: len(str(x).split()))
    df_train["review_len_chars"] = df_train["review_text_str"].apply(lambda x: len(str(x)))

q1 = df_train["review_len_words"].quantile(0.25)
q3 = df_train["review_len_words"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outlier_mask = (df_train["review_len_words"] < lower) | (df_train["review_len_words"] > upper)
outlier_count = int(outlier_mask.sum())

# 1.f (analyze lengths)
print("Review-text word-length outliers:", outlier_count)
print("IQR lower bound:", round(lower, 3))
print("IQR upper bound:", round(upper, 3))

# Save plots for review lengths
save_boxplot(df_train["review_len_words"], "1.f Review Text Lengths (Boxplot)", "word count", "1f_review_length_boxplot.png")
save_hist(df_train["review_len_words"], "1.f Review Text Lengths (Histogram)", "word count", "1f_review_length_hist.png", bins=50)

df_train.loc[outlier_mask, ["ratings", "review_len_words", "review_len_chars", "review_text_str", "summary_str"]] \
    .head(15) \
    .to_csv(os.path.join(OUTPUT_DIR, "1f_review_length_outlier_examples.csv"), index=False)


# %% [markdown]
# ## Text Columns Chosen
# For sentiment analysis, the notebook uses `summary` and `reviewText` together.
# `summary` captures the short headline sentiment, while `reviewText` carries the detailed opinion. Combining them preserves more sentiment signal than using either field alone.

# %%
# 1.g. (check for duplicates)
if "review_text_str" not in df_train.columns:
    df_train["review_text_str"] = df_train["review_text"].apply(to_text_maybe_list)

if "summary_str" not in df_train.columns:
    df_train["summary_str"] = df_train["summary"].apply(to_text_maybe_list)

if "full_text" not in df_train.columns:
    df_train["full_text"] = df_train.apply(
        lambda r: combine_summary_review(r["summary_str"], r["review_text_str"]),
        axis=1
    )
    df_train["full_text"] = df_train["full_text"].fillna("").astype(str)

dup_full_text = df_train.duplicated(subset=["full_text"]).sum()
dup_summary_review = df_train.duplicated(subset=["summary_str", "review_text_str"]).sum()

print("Duplicate full_text:", dup_full_text)
print("Duplicate summary+review:", dup_summary_review)
print("Duplicate full_text percentage:", round(100 * dup_full_text / len(df_train), 3))

# %%
# Labeling
# 2.a.i / 2.a.ii / 2.a.iii (apply rating-based sentiment labels)
df_train["label"] = df_train["ratings"].apply(label_from_rating)
df_train = df_train.dropna(subset=["label"]).copy()

print("Label distribution:")
print(df_train["label"].value_counts())

# %%
# Model-specific preprocessing
# 4 (apply model-specific preprocessing for each selected lexicon)
df_train["text_vader"] = df_train["full_text"].apply(preprocess_for_vader)
df_train["text_blob"]  = df_train["full_text"].apply(preprocess_for_textblob)

# %% [markdown]
# ## Lexicon And Preprocessing Justification
# The notebook keeps VADER and TextBlob as the two Phase 1 lexicon models.
# VADER is designed for short opinion-rich text and benefits from punctuation and emphasis, so its preprocessing is intentionally light. TextBlob uses polarity over cleaned text, so the notebook lowercases and removes most non-letter noise before scoring. SentiWordNet was not chosen here because it requires a heavier word-sense pipeline, which is harder to keep consistent for a compact Phase 1 baseline.

# %%
# Sampling
# 5 (randomly select 1000 reviews from your dataset)
df_s = df_train.sample(
    n=min(SAMPLE_SIZE, len(df_train)),
    random_state=RANDOM_SEED
).copy()

print("Sample size:", len(df_s))
print("Sample label distribution:")
print(df_s["label"].value_counts())

y_true = df_s["label"].astype(str)

# %%
# Predictions
# 6.a (build two lexicon-based sentiment analysis models)
y_vader = vader_predict(df_s["text_vader"])
y_blob  = textblob_predict(df_s["text_blob"])

# %%
# Evaluation
# 7 (accuracy/precision/recall/F1 + confusion matrix)
labels_order = ("Negative", "Neutral", "Positive")

acc_v, pr_v, rc_v, f1_v, cm_v = evaluate(y_true, y_vader, labels=labels_order)
acc_b, pr_b, rc_b, f1_b, cm_b = evaluate(y_true, y_blob, labels=labels_order)

save_confusion(cm_v, labels_order, "VADER Confusion Matrix", "7_cm_vader.png")
save_confusion(cm_b, labels_order, "TextBlob Confusion Matrix", "7_cm_textblob.png")

comparison = pd.DataFrame([
    {"Model": "VADER", "Accuracy": acc_v, "Precision_w": pr_v, "Recall_w": rc_v, "F1_w": f1_v},
    {"Model": "TextBlob", "Accuracy": acc_b, "Precision_w": pr_b, "Recall_w": rc_b, "F1_w": f1_b},
])
comparison = comparison.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

comparison.to_csv(os.path.join(OUTPUT_DIR, "7_lexicon_comparison_table.csv"), index=False)

print(comparison)

# %%
# Accuracy bar chart
plt.figure(figsize=(6, 4))

bars = plt.bar(
    comparison["Model"],
    comparison["Accuracy"],
    color=["red", "blue"]   # VADER = red, TextBlob = blue
)

plt.title("Accuracy Comparison: VADER vs TextBlob")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_accuracy_comparison_chart.png"))
if SHOW_PLOTS:
    plt.show()
plt.close()

# %%
# Metrics comparison (Accuracy / Precision / Recall / F1)
metrics = ["Accuracy", "Precision_w", "Recall_w", "F1_w"]

vader_scores = comparison[comparison["Model"] == "VADER"][metrics].values.flatten()
blob_scores  = comparison[comparison["Model"] == "TextBlob"][metrics].values.flatten()

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, vader_scores, width, label="VADER", color="red")
plt.bar(x + width / 2, blob_scores,  width, label="TextBlob", color="blue")

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.title("VADER vs TextBlob Performance Comparison")
plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1"])
plt.ylim(0, 1)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_metric_comparison_chart.png"))
if SHOW_PLOTS:
    plt.show()
plt.close()

# %% [markdown]
# ## Phase 2: Machine Learning, Comparison, Review Enhancement, and Local LLM Tasks
# 
# This section adds only Phase 2 code and comments under the existing Phase 1 notebook content.
# All Phase 1 code/comments remain unchanged.

# %%
# 11 / 12 / 13 / 14 / 15 / 16 / 17
# Phase 2 setup for outputs and reproducibility (kept separate from Phase 1 config)
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

PHASE2_OUTPUT_DIR = "phase2_outputs"
PHASE2_SAMPLE_SIZE = 2500
PHASE2_TEST_SIZE = 0.30
PHASE2_RANDOM_STATE = 42

ensure_dir(PHASE2_OUTPUT_DIR)
print("Phase 2 output folder:", PHASE2_OUTPUT_DIR)

# 11 / 13 / 14 helper: save confusion matrices only to phase2_outputs
def save_confusion_phase2(cm, labels, title, filename):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PHASE2_OUTPUT_DIR, filename))
    plt.show()
    plt.close()

# %%
# 11.a. Select a subset of minimum 2000 reviews and prepare labels/text
# This cell rebuilds a clean Phase 2 subset (2500 rows) from the original dataset.
phase2_df = load_reviews_dataset(DATASET_PATH).copy()

phase2_df["summary_text"] = phase2_df.get("summary", "").apply(to_text_maybe_list)
phase2_df["review_text_clean"] = phase2_df.get("review_text", "").apply(to_text_maybe_list)
phase2_df["full_text"] = phase2_df.apply(
    lambda row: combine_summary_review(row["summary_text"], row["review_text_clean"]), axis=1
)

phase2_df["ratings"] = pd.to_numeric(phase2_df["ratings"], errors="coerce")
phase2_df["label"] = phase2_df["ratings"].apply(label_from_rating)
phase2_df = phase2_df.dropna(subset=["ratings", "label", "full_text"]).copy()
phase2_df["full_text"] = phase2_df["full_text"].astype(str).str.strip()
phase2_df = phase2_df[phase2_df["full_text"].str.len() > 0].copy()

phase2_subset = phase2_df.sample(n=PHASE2_SAMPLE_SIZE, random_state=PHASE2_RANDOM_STATE).copy()
phase2_subset = phase2_subset.reset_index(drop=True)

print("Phase 2 subset shape:", phase2_subset.shape)
print("Class distribution:\n", phase2_subset["label"].value_counts())

# %% [markdown]
# ### 11.b. Data exploration and preprocessing justification
# 
# We keep preprocessing simple and in module scope:
# - Remove missing/empty records for fair model training.
# - Normalize text with lowercase, URL removal, and whitespace cleanup.
# - Keep labels from rating rules already used in Phase 1.
# - Check class distribution and review length to validate the subset.

# %%
# 11.b. Carry out exploration and preprocessing on the subset
# This cell reports subset quality and builds one cleaned text column for ML.
phase2_subset["review_word_count"] = phase2_subset["full_text"].str.split().apply(len)
print("Review length summary:\n", phase2_subset["review_word_count"].describe())
print("Label proportions:\n", phase2_subset["label"].value_counts(normalize=True).round(4))

# 11.b.i. Simple text normalization for ML pipeline input
phase2_subset["ml_text"] = phase2_subset["full_text"].astype(str).str.lower()
phase2_subset["ml_text"] = phase2_subset["ml_text"].str.replace(r"http\S+|www\.\S+", " ", regex=True)
phase2_subset["ml_text"] = phase2_subset["ml_text"].str.replace(r"\s+", " ", regex=True).str.strip()

print("Prepared ML text rows:", len(phase2_subset))

# %% [markdown]
# ### 11.c. Text representation choice
# 
# We use TF-IDF because it is a suitable and standard module-level representation for sentiment text classification. It converts text to weighted word features, is easy to explain, and works well with Logistic Regression and SVM for this task.

# %%
# 11.c. Represent text using TF-IDF
# Fit/transform will happen inside model pipelines to avoid data leakage.
X_all = phase2_subset["ml_text"].copy()
y_all = phase2_subset["label"].copy()

print("TF-IDF representation selected for model pipelines.")

# %%
# 11.d. Split data into 70% train and 30% test using stratified label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=PHASE2_TEST_SIZE,
    random_state=PHASE2_RANDOM_STATE,
    stratify=y_all
)

print("Train size:", len(X_train), "Test size:", len(X_test))
print("Train label distribution:\n", y_train.value_counts(normalize=True).round(4))
print("Test label distribution:\n", y_test.value_counts(normalize=True).round(4))

# %%
# 11.e. Build two ML sentiment models using training data
# 11.e.i. Model 1: Logistic Regression with TF-IDF
start_lr = time.time()
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000, random_state=PHASE2_RANDOM_STATE))
])
lr_pipeline.fit(X_train, y_train)
lr_train_seconds = time.time() - start_lr

# 11.e.ii. Model 2: SVM (LinearSVC) with TF-IDF
start_svm = time.time()
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])
svm_pipeline.fit(X_train, y_train)
svm_train_seconds = time.time() - start_svm

# 12. Record training process results
training_summary_df = pd.DataFrame([
    {"model": "LogisticRegression", "train_rows": len(X_train), "train_seconds": round(lr_train_seconds, 3)},
    {"model": "LinearSVC", "train_rows": len(X_train), "train_seconds": round(svm_train_seconds, 3)}
])
training_summary_path = os.path.join(PHASE2_OUTPUT_DIR, "12_training_summary.csv")
training_summary_df.to_csv(training_summary_path, index=False)
training_summary_df

# %%
# 13. Test both ML models using the same 30% test data and report required metrics
labels_order = ["Negative", "Neutral", "Positive"]

y_pred_lr = lr_pipeline.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)

acc_lr, pr_lr, rc_lr, f1_lr, cm_lr = evaluate(y_test, y_pred_lr, labels=labels_order)
acc_svm, pr_svm, rc_svm, f1_svm, cm_svm = evaluate(y_test, y_pred_svm, labels=labels_order)

ml_test_metrics_df = pd.DataFrame([
    {"model": "LogisticRegression", "accuracy": acc_lr, "precision_w": pr_lr, "recall_w": rc_lr, "f1_w": f1_lr},
    {"model": "LinearSVC", "accuracy": acc_svm, "precision_w": pr_svm, "recall_w": rc_svm, "f1_w": f1_svm}
])
ml_test_metrics_path = os.path.join(PHASE2_OUTPUT_DIR, "13_ml_test_metrics.csv")
ml_test_metrics_df.to_csv(ml_test_metrics_path, index=False)

save_confusion_phase2(cm_lr, labels_order, "13 Confusion Matrix - Logistic Regression", "13_cm_logistic_regression.png")
save_confusion_phase2(cm_svm, labels_order, "13 Confusion Matrix - Linear SVM", "13_cm_linear_svm.png")

ml_test_metrics_df

# %%
# 14.a. Prepare fair comparison data (apples-to-apples): exact same test set for Lexicon and ML models
# 14.b. Run Lexicon and ML models on same test rows and compare with same metrics
X_test_vader = X_test.apply(preprocess_for_vader)
X_test_textblob = X_test.apply(preprocess_for_textblob)

y_pred_vader_same = vader_predict(X_test_vader)
y_pred_textblob_same = textblob_predict(X_test_textblob)

acc_v, pr_v, rc_v, f1_v, cm_v = evaluate(y_test, y_pred_vader_same, labels=labels_order)
acc_tb, pr_tb, rc_tb, f1_tb, cm_tb = evaluate(y_test, y_pred_textblob_same, labels=labels_order)

same_test_comparison_df = pd.DataFrame([
    {"model": "VADER", "accuracy": acc_v, "precision_w": pr_v, "recall_w": rc_v, "f1_w": f1_v},
    {"model": "TextBlob", "accuracy": acc_tb, "precision_w": pr_tb, "recall_w": rc_tb, "f1_w": f1_tb},
    {"model": "LogisticRegression", "accuracy": acc_lr, "precision_w": pr_lr, "recall_w": rc_lr, "f1_w": f1_lr},
    {"model": "LinearSVC", "accuracy": acc_svm, "precision_w": pr_svm, "recall_w": rc_svm, "f1_w": f1_svm}
])

same_test_comparison_path = os.path.join(PHASE2_OUTPUT_DIR, "14_model_comparison_same_test.csv")
same_test_comparison_df.to_csv(same_test_comparison_path, index=False)

save_confusion_phase2(cm_v, labels_order, "14 Confusion Matrix - VADER (same test)", "14_cm_vader_same_test.png")
save_confusion_phase2(cm_tb, labels_order, "14 Confusion Matrix - TextBlob (same test)", "14_cm_textblob_same_test.png")

same_test_comparison_df.sort_values("accuracy", ascending=False)

# %% [markdown]
# ### 15.a, 15.b, 15.c Review-based rating enhancement (from review sentiment)
# 
# Chosen strategy: improve raw rating values by adding review sentiment signal from a trained model.
# 
# Pseudo-code:
# 1. Predict sentiment class from review text.
# 2. Convert sentiment to rating adjustment: Positive = +1, Neutral = 0, Negative = -1.
# 3. Add adjustment to original rating and clip to valid range [1, 5].
# 4. Compare original vs enhanced rating distributions and agreement with sentiment labels.

# %%
# 15.a. Explain enhancement: use review sentiment to refine rating value signal
# 15.b. Pseudo-code is documented in the markdown cell above
# 15.c. Implement enhancement and record results
sentiment_to_delta = {"Positive": 1, "Neutral": 0, "Negative": -1}

phase2_test_results = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "pred_label_lr": y_pred_lr
})

phase2_test_results = phase2_test_results.merge(
    phase2_subset[["ml_text", "ratings"]].rename(columns={"ml_text": "text"}),
    on="text",
    how="left"
)

phase2_test_results["delta"] = phase2_test_results["pred_label_lr"].map(sentiment_to_delta).fillna(0)
phase2_test_results["enhanced_rating"] = (phase2_test_results["ratings"] + phase2_test_results["delta"]).clip(1, 5)

enhancement_summary_df = pd.DataFrame([
    {"metric": "original_mean_rating", "value": round(float(phase2_test_results["ratings"].mean()), 4)},
    {"metric": "enhanced_mean_rating", "value": round(float(phase2_test_results["enhanced_rating"].mean()), 4)},
    {"metric": "num_test_rows", "value": int(len(phase2_test_results))}
])

enhancement_preview_path = os.path.join(PHASE2_OUTPUT_DIR, "15_enhanced_ratings_preview.csv")
phase2_test_results[["text", "ratings", "pred_label_lr", "delta", "enhanced_rating"]].head(50).to_csv(enhancement_preview_path, index=False)

enhancement_summary_path = os.path.join(PHASE2_OUTPUT_DIR, "15_enhancement_summary.csv")
enhancement_summary_df.to_csv(enhancement_summary_path, index=False)

enhancement_summary_df

# %%
# 16 / 17 Local Hugging Face setup (host models locally)
# Steps 16 and 17 require transformers, torch, and sentencepiece to be installed from requirements.txt.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Shared helper for local text generation with seq2seq models (T5/FLAN-T5)
def local_seq2seq_generate(model_name, prompt, max_new_tokens=120):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# %%
# 16. Summarize 10 reviews longer than 100 words into 50 words using local HF model
long_reviews_df = phase2_subset[phase2_subset["review_word_count"] > 100].copy()
long_reviews_df = long_reviews_df.head(10).copy()

def to_50_words(text):
    words = str(text).split()
    return " ".join(words[:50])

summary_rows = []
for idx, row in long_reviews_df.iterrows():
    prompt = "summarize in about 50 words: " + str(row["full_text"])
    generated = local_seq2seq_generate("t5-small", prompt, max_new_tokens=120)
    summary_50 = to_50_words(generated)
    summary_rows.append({
        "source_index": int(idx),
        "original_review": row["full_text"],
        "summary_50_words": summary_50
    })

summary_10_df = pd.DataFrame(summary_rows)
summary_10_path = os.path.join(PHASE2_OUTPUT_DIR, "16_long_reviews_50word_summaries.csv")
summary_10_df.to_csv(summary_10_path, index=False)

print("First two summaries for report:")
summary_10_df.head(2)

# %%
# 17. Generate one service representative response for a question-like review using local HF model
question_reviews_df = phase2_subset[phase2_subset["full_text"].str.contains("\?", regex=True, na=False)].copy()

if len(question_reviews_df) == 0:
    question_text = "Can you help me with the product size and return policy?"
else:
    question_text = str(question_reviews_df.iloc[0]["full_text"])

response_prompt = (
    "You are a customer service representative. "
    "Write a helpful, polite response to this customer review question: " + question_text
)

service_response = local_seq2seq_generate("google/flan-t5-small", response_prompt, max_new_tokens=120)

step17_df = pd.DataFrame([
    {"question_review": question_text, "service_response": service_response}
])
step17_path = os.path.join(PHASE2_OUTPUT_DIR, "17_question_response.csv")
step17_df.to_csv(step17_path, index=False)

step17_df


