import re

import pandas as pd


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