import pandas as pd
import string
from pathlib import Path

"""
Script creates a "bilingual" language by shifting the each letter by ASCII value designated by SHIFT size.
All shifted words have the same concept_id as their English counterparts to map to same semantic feature

# ONsize, frequency, and concreteness are left unchanged for now
"""


SHIFT = 13 # Change Shift Size

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "helper_txt_files" / "threshold2_df_fast.csv"
OUTPUT_CSV = ROOT / "helper_txt_files" / "threshold2_df_fast_bilingual13.csv"

alphabet = string.ascii_lowercase
alpha_index = {ch: i for i, ch in enumerate(alphabet)}



def caesar_shift_word(word: str, shift: int = SHIFT) -> str:
    shifted = []
    for ch in word:
        if ch not in alpha_index:
            raise ValueError(f"Unsupported character '{ch}' in word '{word}'")
        new_idx = (alpha_index[ch] + shift) % 26
        shifted.append(alphabet[new_idx])
    return "".join(shifted)

def main():
    df = pd.read_csv(INPUT_CSV)

    if "Word" not in df.columns:
        raise ValueError("Expected a 'Word' column in lexicon_characteristics.csv")

    english = df.copy()
    english["language"] = "en"

    # set concept_id to match counterpart
    # NOW: concept_id = original word
    english["concept_id"] = english["Word"].str.upper()

    shifted = df.copy()
    shifted["Word"] = shifted["Word"].apply(caesar_shift_word)
    shifted["language"] = f"c{SHIFT}"
    shifted["concept_id"] = df["Word"].str.upper() # set concept ID to match counterpart

    bilingual = pd.concat([english, shifted], ignore_index=True)

    bilingual.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved bilingual lexicon to: {OUTPUT_CSV}")
    print(f"Original rows: {len(df)}")
    print(f"Bilingual rows: {len(bilingual)}")

if __name__ == "__main__":
    main()