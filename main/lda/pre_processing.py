import swifter
import pandas as pd

def extract_pos_ds(df: pd.DataFrame, pos_affix: str, store_path: str) -> pd.DataFrame:
    df = df.swifter.apply(lambda x: x.split(' '))
    df = df.swifter.apply(lambda x: [w.split(pos_affix)[0] for w in x if w.endswith(pos_affix)])
    df = df[df.map(len) > 1].map(lambda x: ' '.join(x)).drop_duplicates()

    # I like the inline js type of condition expression execution
    store_path is not None and df.to_csv(store_path, index=False)

    return df
