import pandas as pd

def remove_na_titles(
        df, 
        title_column: str
        ):
    return df.dropna(
            subset = title_column, 
            axis = 0
            )

def text_in_lower_case(
            df, 
            text_column):
        # Make  lower case and remove double spaces to avoid false duplicates
        return df[text_column].str.lower().str.replace('  ', ' ')

def aggregate_df(
            df,
            group_by_columns,
            col_to_aggregate,
            operation,
            new_name_for_col_aggregated):
      if operation == 'average':
          return pd.DataFrame(
               df.groupby(group_by_columns, dropna=False)[col_to_aggregate].mean()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})
      if operation == 'median':
          return pd.DataFrame(
               df.groupby(group_by_columns, dropna=False)[col_to_aggregate].median()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})
      if operation == 'count':
          return pd.DataFrame(
               df.groupby(group_by_columns, dropna=False)[col_to_aggregate].count()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})
      if operation == 'min':
          return pd.DataFrame(
               df.groupby(group_by_columns, dropna=False)[col_to_aggregate].min()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})
      if operation == 'max':
          return pd.DataFrame(
               df.groupby(group_by_columns, dropna=False)[col_to_aggregate].max()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})


def weighted_rating(
    titles_df,
    n_of_reviews_column,
    avg_rating_column,
    m,
    C,
    quantile_for_min_votes
    ):

    # Calculate minimum number of votes (m)
    q = quantile_for_min_votes
    m = titles_df[n_of_reviews_column].quantile(q)

    # Calculate mean vote across the whole dataset
    C = titles_df[avg_rating_column].mean()

    v = titles_df[n_of_reviews_column]
    R = titles_df[avg_rating_column]
    return (v*R/(v+m) + m*C/(v+m))

def review_time_conversion(date):
    return pd.to_datetime(date, unit='s')