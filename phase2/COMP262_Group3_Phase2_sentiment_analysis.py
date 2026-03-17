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


