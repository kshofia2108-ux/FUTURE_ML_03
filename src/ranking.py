import pandas as pd

def rank_candidates(results):

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["similarity_score", "skill_score"],
        ascending=False
    )

    return df