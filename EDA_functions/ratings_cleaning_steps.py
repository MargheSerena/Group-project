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
               df.groupby(group_by_columns)[col_to_aggregate].mean()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})
      if operation == 'median':
          return pd.DataFrame(
               df.groupby(group_by_columns)[col_to_aggregate].median()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})
      if operation == 'count':
          return pd.DataFrame(
               df.groupby(group_by_columns)[col_to_aggregate].count()
                    ).reset_index().rename(
                         columns = {col_to_aggregate: new_name_for_col_aggregated})