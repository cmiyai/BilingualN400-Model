import pandas as pd

path = "./helper_txt_files/threshold2_df_fast_bilingual.csv"
df = pd.read_csv(path)
l2 = df["language"] != "en"
df.loc[l2, "Word"] = df.loc[l2, "Word"].replace({
    "ngax": "ngai",
    "grax": "graf",
    "spex": "spec",
})
df.to_csv(path, index=False)