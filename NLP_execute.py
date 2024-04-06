from NLP_preprocessing import add_tokens_to_description, create_tokens, add_TFIDF_tSVD_to_df
import pandas as pd
import numpy as np

def main():
    # Define the dataframe and the train/ test indices
    df = pd.read_csv("English_fiction_pre_PCA_3.csv")
    train_df = pd.read_csv("original_data/train_indices.csv")
    test_df = pd.read_csv("original_data/test_indices.csv")
    
    train_index = train_df["index"].tolist()
    test_index = test_df["index"].tolist()
    n_tSVD_components = 3000
    
    # Add tokens from other columns to the description column, specifically author, title, and publisher
    add_tokens_to_description(df)
    
    # Create tokens from the book descriptions and save this in a new column called "tokens"
    create_tokens(df, "description", "tokens")
    
    # Apply truncated SVD to the tokens column and append the tSVD components to the Xtrain and Xtest dataframe:
    tSVD_model, tSVD_components_X_train, tSVD_components_X_test, X_train_df, X_test_df = add_TFIDF_tSVD_to_df(df, "tokens", n_tSVD_components, train_index, test_index)

    X_train_df.to_csv(
    path_or_buf = "X_train_tSVD.csv",
    index = False
    )

    X_test_df.to_csv(
    path_or_buf = "X_test_tSVD.csv",
    index = False
    )

    return tSVD_model, tSVD_components_X_train, tSVD_components_X_test, X_train_df, X_test_df

if __name__ == "__main__":
    main()