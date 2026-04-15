import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


FIRST_KEYWORDS = [
    "during your standby phase",
    "during each of your standby phases",
    "if this card is activated",
    "add 1",
    "set 1",
    "field zone",
    "if you control no monsters",
    "you can only activate 1",
]

SECOND_KEYWORDS = [
    "during your opponent's turn",
    "when an opponent's monster declares an attack",
    "if your opponent",
    "your opponent controls",
    "opponent's turn",
    "quick effect",
]

TAG_PATTERNS = {
    "negate": ["negate"],
    "destroy": ["destroy", "destroyed"],
    "draw": ["draw"],
    "banish": ["banish", "banished"],
}


def clean_text(value: str) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def make_combined_text(df: pd.DataFrame) -> pd.Series:
    name_col = "name_official" if "name_official" in df.columns else "name"
    pieces = [
        df[name_col].fillna(""),
        df["description"].fillna(""),
        df.get("type", pd.Series([""] * len(df))),
        df.get("sub_type", pd.Series([""] * len(df))),
        df.get("attribute", pd.Series([""] * len(df))),
    ]
    combined = pieces[0].astype(str)
    for p in pieces[1:]:
        combined = combined + " " + p.astype(str)
    return combined.apply(clean_text)


def infer_first_second_label(description: str) -> str:
    text = clean_text(description)
    first_score = sum(1 for kw in FIRST_KEYWORDS if kw in text)
    second_score = sum(1 for kw in SECOND_KEYWORDS if kw in text)

    if second_score > first_score:
        return "second"
    if first_score > second_score:
        return "first"

    # Fallback heuristics when scores are tied.
    if "opponent" in text:
        return "second"
    return "first"


def infer_effect_tag(description: str) -> str:
    text = clean_text(description)

    # If several tags appear, choose the earliest mention in text.
    best_tag = "other"
    best_pos = None

    for tag, keywords in TAG_PATTERNS.items():
        for kw in keywords:
            pos = text.find(kw)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_tag = tag

    return best_tag


def deduplicate_cards(df: pd.DataFrame) -> pd.DataFrame:
    name_col = "name_official" if "name_official" in df.columns else "name"
    df = df.copy()
    df["_name_norm"] = df[name_col].apply(clean_text)
    df["_desc_norm"] = df["description"].apply(clean_text)
    deduped = df.drop_duplicates(subset=["_name_norm", "_desc_norm"], keep="first").copy()
    deduped = deduped.drop(columns=["_name_norm", "_desc_norm"])
    return deduped


def train_logistic_first_second(df: pd.DataFrame, random_state: int = 42) -> dict:
    data = df.copy()
    data["first_second_label"] = data["description"].apply(infer_first_second_label)
    data["combined_text"] = make_combined_text(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data["combined_text"],
        data["first_second_label"],
        test_size=0.2,
        random_state=random_state,
        stratify=data["first_second_label"],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, digits=4)
    acc = accuracy_score(y_test, preds)

    return {
        "model": model,
        "accuracy": acc,
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def train_svm_effect_tag(df: pd.DataFrame, random_state: int = 42) -> dict:
    data = df.copy()
    data["effect_tag"] = data["description"].apply(infer_effect_tag)
    data["combined_text"] = make_combined_text(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data["combined_text"],
        data["effect_tag"],
        test_size=0.2,
        random_state=random_state,
        stratify=data["effect_tag"],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
            ("clf", LinearSVC()),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, digits=4)
    acc = accuracy_score(y_test, preds)

    return {
        "model": model,
        "accuracy": acc,
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YuGiOh ML models: Logistic Regression and SVM."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="YuGiOh Dataset.csv",
        help="Path ke file dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Folder output untuk menyimpan model dan metrik.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "description" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'description'.")

    before = len(df)
    df = deduplicate_cards(df)
    after = len(df)

    lr_result = train_logistic_first_second(df)
    svm_result = train_svm_effect_tag(df)

    joblib.dump(lr_result["model"], out_dir / "logreg_first_second.joblib")
    joblib.dump(svm_result["model"], out_dir / "svm_effect_tag.joblib")

    metrics_text = []
    metrics_text.append("=== Dataset Summary ===")
    metrics_text.append(f"Rows awal: {before}")
    metrics_text.append(f"Rows setelah deduplikasi (nama + efek sama): {after}")
    metrics_text.append("")

    metrics_text.append("=== Logistic Regression: First vs Second ===")
    metrics_text.append(f"Train size: {lr_result['train_size']}")
    metrics_text.append(f"Test size: {lr_result['test_size']}")
    metrics_text.append(f"Accuracy: {lr_result['accuracy']:.4f}")
    metrics_text.append(lr_result["report"])
    metrics_text.append("")

    metrics_text.append("=== SVM: Effect Tag Classification ===")
    metrics_text.append(f"Train size: {svm_result['train_size']}")
    metrics_text.append(f"Test size: {svm_result['test_size']}")
    metrics_text.append(f"Accuracy: {svm_result['accuracy']:.4f}")
    metrics_text.append(svm_result["report"])

    metrics_output = "\n".join(metrics_text)
    (out_dir / "training_metrics.txt").write_text(metrics_output, encoding="utf-8")

    print(metrics_output)
    print(f"\nModel tersimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
