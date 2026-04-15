#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess_yugioh import (
    deduplicate_cards,
    infer_effect_tag,
    infer_first_second_label,
    make_combined_text,
)


def train_logistic_first_second(df: pd.DataFrame, random_state: int = 42, n_splits: int = 5) -> dict:
    data = df.copy()
    data["first_second_label"] = data["description"].apply(infer_first_second_label)
    data["combined_text"] = make_combined_text(data)

    X = data["combined_text"]
    y = data["first_second_label"]

    # Build pipeline
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )

    # K-Fold Cross Validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

    # Train final model on entire dataset
    model.fit(X, y)

    # For final report, do a simple train-test split for detailed classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model_for_report = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )
    model_for_report.fit(X_train, y_train)
    preds = model_for_report.predict(X_test)
    report = classification_report(y_test, preds, digits=4)

    return {
        "model": model,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "report": report,
        "total_samples": len(X),
    }


def train_svm_effect_tag(df: pd.DataFrame, random_state: int = 42, n_splits: int = 5) -> dict:
    data = df.copy()
    data["effect_tag"] = data["description"].apply(infer_effect_tag)
    data["combined_text"] = make_combined_text(data)

    X = data["combined_text"]
    y = data["effect_tag"]

    # Build pipeline
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
            ("clf", LinearSVC(max_iter=2000)),
        ]
    )

    # K-Fold Cross Validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

    # Train final model on entire dataset
    model.fit(X, y)

    # For final report, do a simple train-test split for detailed classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model_for_report = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
            ("clf", LinearSVC(max_iter=2000)),
        ]
    )
    model_for_report.fit(X_train, y_train)
    preds = model_for_report.predict(X_test)
    report = classification_report(y_test, preds, digits=4)

    return {
        "model": model,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "report": report,
        "total_samples": len(X),
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

    metrics_text.append("=== Logistic Regression: First vs Second (5-Fold CV) ===")
    metrics_text.append(f"Total samples: {lr_result['total_samples']}")
    metrics_text.append(f"K-Fold scores: {[f'{s:.4f}' for s in lr_result['cv_scores']]}")
    metrics_text.append(f"Mean Accuracy (CV): {lr_result['cv_mean']:.4f} (+/- {lr_result['cv_std']:.4f})")
    metrics_text.append("")
    metrics_text.append("Classification Report (on 80/20 split for reference):")
    metrics_text.append(lr_result["report"])
    metrics_text.append("")

    metrics_text.append("=== SVM: Effect Tag Classification (5-Fold CV) ===")
    metrics_text.append(f"Total samples: {svm_result['total_samples']}")
    metrics_text.append(f"K-Fold scores: {[f'{s:.4f}' for s in svm_result['cv_scores']]}")
    metrics_text.append(f"Mean Accuracy (CV): {svm_result['cv_mean']:.4f} (+/- {svm_result['cv_std']:.4f})")
    metrics_text.append("")
    metrics_text.append("Classification Report (on 80/20 split for reference):")
    metrics_text.append(svm_result["report"])

    metrics_output = "\n".join(metrics_text)
    (out_dir / "training_metrics.txt").write_text(metrics_output, encoding="utf-8")

    print(metrics_output)
    print(f"\nModel tersimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
