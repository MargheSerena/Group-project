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
        # Make  lower case to avoid false duplicates
        return df[text_column].str.lower()

            