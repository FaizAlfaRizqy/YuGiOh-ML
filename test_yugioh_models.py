#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocess_yugioh import make_combined_text


def build_input_text(card_name: str, card_effect: str) -> str:
    row = {
        "name_official": card_name,
        "description": card_effect,
        "type": "",
        "sub_type": "",
        "attribute": "",
    }
    df = pd.DataFrame([row])
    return make_combined_text(df).iloc[0]


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. Jalankan train_yugioh_models.py dulu."
        )
    return joblib.load(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test model YuGiOh: prediksi first/second dan tag efek kartu."
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Nama kartu.",
    )
    parser.add_argument(
        "--effect",
        type=str,
        help="Efek/deskripsi kartu.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Folder model hasil training.",
    )
    args = parser.parse_args()

    card_name = args.name or input("Masukkan nama kartu: ").strip()
    card_effect = args.effect or input("Masukkan efek kartu: ").strip()

    if not card_name:
        raise ValueError("Nama kartu tidak boleh kosong.")
    if not card_effect:
        raise ValueError("Efek kartu tidak boleh kosong.")

    models_dir = Path(args.models_dir)
    logreg_path = models_dir / "logreg_first_second.joblib"
    svm_path = models_dir / "svm_effect_tag.joblib"

    logreg_model = load_model(logreg_path)
    svm_model = load_model(svm_path)

    input_text = build_input_text(card_name, card_effect)

    turn_pred = logreg_model.predict([input_text])[0]
    tag_pred = svm_model.predict([input_text])[0]

    print("\n=== Hasil Prediksi ===")
    print(f"Nama Kartu       : {card_name}")
    print(f"First/Second     : {turn_pred}")
    print(f"Tag Efek         : {tag_pred}")


if __name__ == "__main__":
    main()
