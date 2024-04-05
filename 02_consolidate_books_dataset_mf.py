import time
import pandas as pd
from NLP_preprocessing import identify_language, create_BOW_feature_for_english_descriptions, add_TFIDF_tSVD_to_df

def main():
    clean_df = pd.read_csv("English_fiction_pre_PCA_3.csv")
    print(clean_df.size)
    # clean_df = clean_df[:10000]
    start = time.perf_counter()
    create_BOW_feature_for_english_descriptions(clean_df, "description", "english_BOW")
    after_bow = time.perf_counter()
    tSVD_model, tSVD_components = add_TFIDF_tSVD_to_df(clean_df, "english_BOW", 30)
    end = time.perf_counter()
    print("tSVD_model.explained_variance_ratio_", tSVD_model.explained_variance_ratio_)
    print("tSVD_model.explained_variance_ratio_.sum()", tSVD_model.explained_variance_ratio_.sum())
    print("tSVD_model.components_.shape", tSVD_model.components_.shape)
    print(f"BOW took: {after_bow - start}s")
    print(f"SVD took: {end - after_bow}s")

if __name__ == "__main__":
    main()
