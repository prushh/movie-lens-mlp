from typing import List, Dict, Any

import pandas as pd


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    return df


def drop(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df.drop(columns=columns, inplace=True)
    return df


def fill_na(df: pd.DataFrame, column: str, use_median: bool) -> pd.DataFrame:
    if use_median:
        median = df[column].median()
        df[column].fillna(median, inplace=True)
        return df
    mean = df[column].mean()
    df[column].fillna(mean, inplace=True)
    return df


def convert_to(df: pd.DataFrame, column: str, _type: str) -> pd.DataFrame:
    df[column] = df[column].astype(_type)
    return df


def rename(df: pd.DataFrame, _dict: Dict[str, str]) -> pd.DataFrame:
    df.rename(columns=_dict, inplace=True)
    return df


def replace(df: pd.DataFrame, column: str, regex: str, value: Any):
    df[column].replace(regex=regex, value=value, inplace=True)
    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(inplace=True)
    return df


def extract_stat_feature(df: pd.DataFrame, by: List[str], column: str, stat: List[str]) -> pd.DataFrame:
    df_stat = pd.DataFrame(
        df.groupby(by=by, as_index=False)[column].agg(stat)
    )
    return df_stat


def apply(df: pd.DataFrame, column: str, func) -> pd.DataFrame:
    df[column] = df[column].apply(func)
    return df


def merge(df_left: pd.DataFrame, df_right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = pd.merge(df_left, df_right, **kwargs)
    return df
