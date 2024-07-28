import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    df, test_size=0.2, val_size=0.1, random_state=42, stratify=None
):
    train_df, test_plus_val_df = train_test_split(
        df,
        test_size=(test_size + val_size), 
        random_state=random_state, 
        stratify=stratify
    )
    if val_size > 0:
        val_test_ratio = val_size/(val_size + test_size)
        validation_df, test_df = train_test_split(
            test_plus_val_df,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=stratify
        )
    else:
        validation_df = pd.DataFrame()
        test_df = test_plus_val_df
    return train_df, validation_df, test_df