from typing import List, Dict, Any

import pandas as pd


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper to DataFrame.dropna() function
    :param df: DataFrame used in the pipe
    :return: the same DataFrame without NaN values
    """
    df.dropna(inplace=True)
    return df


def drop(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Wrapper to DataFrame.drop() function
    :param df: DataFrame used in the pipe
    :param columns: the list of columns to be dropped
    :return: the same DataFrame without the specified columns
    """
    df.drop(columns=columns, inplace=True)
    return df


def fill_na(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    """
    Wrapper to Series.fillna() function
    :param df: DataFrame used in the pipe
    :param column: the column to fill with the specified method
    :param method: the string that define the fill method (median, mean or zero)
    :return: the same DataFrame without NaN values
    """
    if method == 'median':
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'zero':
        df[column].fillna(0, inplace=True)

    return df


def convert_to(df: pd.DataFrame, column: str, _type: str) -> pd.DataFrame:
    """
    Wrapper to Series.astype() function
    :param df: DataFrame used in the pipe
    :param column: the column which dtype will be converted
    :param _type: the new type for the Series specified
    :return: the same DataFrame with new type on specified column
    """
    df[column] = df[column].astype(_type)
    return df


def rename(df: pd.DataFrame, _dict: Dict[str, str]) -> pd.DataFrame:
    """
    Wrapper to DataFrame.rename() function
    :param df: DataFrame used in the pipe
    :param _dict: the dict that specify the couples 'column_name': 'new_column_name'
    :return: the same DataFrame with new name for some columns
    """
    df.rename(columns=_dict, inplace=True)
    return df


def replace(df: pd.DataFrame, column: str, regex: str, value: Any):
    """
    Wrapper to Series.replace() function
    :param df: DataFrame used in the pipe
    :param column: the column which values will be replaced
    :param regex: how to find the values that will be replaced
    :param value: the value to replace any values matching to_replace with
    :return: the same DataFrame with value where to_replace match
    """
    df[column].replace(regex=regex, value=value, inplace=True)
    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper to DataFrame.reset_index() function
    :param df: DataFrame used in the pipe
    :return: the same DataFrame with reset index applied
    """
    df.reset_index(inplace=True)
    return df


def extract_stat_feature(df: pd.DataFrame, by: List[str], column: str, stat: List[str]) -> pd.DataFrame:
    """
    Wrapper to DataFrameGroupBy.aggregate() function
    :param df: DataFrame used in the pipe
    :param by: used to determine the groups for the groupby
    :param column: the column useful to take the specific series where apply aggregate functions
    :param stat: function to use for aggregating the data
    :return: the Dataframe that contains one column for each stat specified
    """
    df_stat = df.groupby(by=by, as_index=False)[column].agg(stat)
    return df_stat
