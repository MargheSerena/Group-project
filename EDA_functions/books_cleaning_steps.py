import pandas as pd
import re


# Extract year if available
def datetime_conversion(value):

    # Convert date into datetime where format is "%Y-%m-%d"
    date = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")

    if not pd.isna(date):
        return date

    else:
        # Convert date into datetime where format is "%Y-%m"
        date = pd.to_datetime(value, format="%Y-%m", errors="coerce")

        if not pd.isna(date):
            return date

        else:
            # Convert date into datetime where format is "%Y"
            date = pd.to_datetime(value, format="%Y", errors="coerce")

            if not pd.isna(date):
                return date

            else:
                return None


# this function removes NaN values
def clean_based_on_date(df, publishedDate_column: str):

    # Filter out NaN and dates containing '?'
    filtered_df = df[(df[publishedDate_column].notna())]

    return filtered_df


# this function converts to datetime
# removes NaN values
# and removes any dates containing a '?'


def clean_date(df, publishedDate_column: str):
    # Convert 'dates' column to datetime
    df[publishedDate_column] = pd.to_datetime(df[publishedDate_column], errors="coerce")

    # Filter out NaN and dates containing '?'
    filtered_df = df[
        df[publishedDate_column].notna()
        & ~df[publishedDate_column].astype(str).str.contains("\?", na=False)
    ]
    return filtered_df
