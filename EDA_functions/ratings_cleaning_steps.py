def remove_na_titles(
        df, 
        title_column: str
        ):
    return df.dropna(
            subset = title_column, 
            axis = 0
            )
            