import os
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import json


def read_csv(data_path: Path) -> pd.DataFrame:
    # Trying , and ; as delimiters.
    try:
        df = pd.read_csv(data_path, index_col=0)
    except:
        try:
            df = pd.read_csv(data_path, delimiter=";", index_col=0)
        except Exception as exc:
            raise RuntimeError(exc)
    # Hack for user study csvs, remove NaN in the "id" column (there are explanation cells).
    # if id in df.columns:
    #     df = df[~df.id.isna()]
    return df


def get_concept_score(data_path: Path, out_name):
    df = read_csv(data_path)

    scores = []
    for l in [1,2,3]:
        print(f"lambda: {l}")
        responses = df.loc[df["lambda"] == l, "q0_llm_answer"].tolist()
        print(responses)
        score = sum([r == 'Yes' for r in  responses])/len(responses)
        scores.append(score)
        print(f"score: {score}")

    results_df = pd.DataFrame(
            data=scores,
            columns=[  # cfg.columns +
                "score",
            ]
        )
    dfs_out = (
        [
            df,
        ]
    )
    results_final = pd.concat(
        dfs_out + [results_df],
        axis=1,
    )
    output_path = "."
    if output_path is not None:
        filename = Path(output_path) / (out_name + "score_eval.csv")
        results_final.to_csv(filename)
        print(f"Saved results in {filename}")

def main() -> None:
    get_concept_score("0_vague_shot_eval.csv", "vague")


if __name__ == "__main__":
    main()