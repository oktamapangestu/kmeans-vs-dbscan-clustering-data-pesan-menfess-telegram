from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class PreparedTextDataset:
    df: pd.DataFrame
    raw_texts: list[str]
    cleaned_texts: list[str]
    cleaned_texts_embed: list[str]
    total_rows: int
    dedup_removed: int


def prepare_text_dataset(
    *,
    input_path: str,
    text_col: str,
    min_len: int,
    min_words: int,
    limit: int,
    limit_random: bool,
    drop_duplicate_texts: bool,
    random_state: int,
    clean_col: str,
    clean_text: Callable[[str], str],
    clean_text_embed: Callable[[str], str] | None = None,
) -> PreparedTextDataset:
    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise SystemExit(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    raw = df[text_col].fillna("").astype(str)
    cleaned = raw.map(clean_text)
    word_counts = cleaned.str.split().str.len()
    keep = (cleaned.str.len() >= min_len) & (word_counts >= min_words)

    filtered_df = df.loc[keep].copy()
    raw_texts = raw.loc[keep].tolist()
    cleaned_texts = cleaned.loc[keep].tolist()

    dedup_removed = 0
    if drop_duplicate_texts:
        dedup_mask = ~pd.Series(cleaned_texts).duplicated(keep="first")
        dedup_removed = int((~dedup_mask).sum())
        dedup_mask_np = dedup_mask.to_numpy()
        filtered_df = filtered_df.loc[filtered_df.index[dedup_mask_np]].copy()
        raw_texts = [text for text, keep_row in zip(raw_texts, dedup_mask_np) if keep_row]
        cleaned_texts = [text for text, keep_row in zip(cleaned_texts, dedup_mask_np) if keep_row]

    if limit > 0:
        if limit_random:
            sampled = filtered_df.assign(_raw_text=raw_texts, _cleaned_text=cleaned_texts).sample(
                n=min(limit, len(filtered_df)),
                random_state=random_state,
            )
            raw_texts = sampled.pop("_raw_text").tolist()
            cleaned_texts = sampled.pop("_cleaned_text").tolist()
            filtered_df = sampled.copy()
        else:
            filtered_df = filtered_df.iloc[:limit].copy()
            raw_texts = raw_texts[:limit]
            cleaned_texts = cleaned_texts[:limit]

    if clean_col:
        filtered_df[clean_col] = cleaned_texts

    if clean_text_embed is not None:
        cleaned_texts_embed = [clean_text_embed(t) for t in raw_texts]
    else:
        cleaned_texts_embed = list(cleaned_texts)

    return PreparedTextDataset(
        df=filtered_df,
        raw_texts=raw_texts,
        cleaned_texts=cleaned_texts,
        cleaned_texts_embed=cleaned_texts_embed,
        total_rows=len(df),
        dedup_removed=dedup_removed,
    )
