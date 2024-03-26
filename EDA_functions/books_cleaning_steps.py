# this function converts to datetime
# removes NaN values
# and removes any dates containing a '?'

def clean_date(
    df,
    publishedDate_column: str    
):
    # Convert 'dates' column to datetime
    df[publishedDate_column] = pd.to_datetime(df[publishedDate_column], errors='coerce')

    # Filter out NaN and dates containing '?'
    filtered_df = df[df[publishedDate_column].notna() & ~df[publishedDate_column].astype(str).str.contains('\?', na=False)]
    return filtered_df


