import pandas as pd
print("pandas", dir(pd))
from langdetect import detect, DetectorFactory
from iso639 import Lang
from rake_nltk import Rake

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

def identify_language(text: str) -> str:
    """Returns the language of a string"""

    # DetectorFactory.seed = 0 means that we will get consistent language 
    # predictions for any edge cases that are ambiguous
    DetectorFactory.seed = 0 
    
    if type(text) == str: 
        try:
            language = detect(text)
            return Lang(language).name
        except:
            return "unknown"
    else:
        return "error"

def calculate_ngrams_RAKE(text: str):
    """ generates a list of n-grams for a given input text using RAKE
    """

    r_unigram = Rake()
    r_phrase = Rake(min_length=2, max_length=3)
    r_unigram.extract_keywords_from_text(text)
    r_phrase.extract_keywords_from_text(text)
    
    keyword_dict_scores = r_unigram.get_word_degrees()
    words = list(keyword_dict_scores.keys())
    words
    
    n_grams = r_phrase.get_ranked_phrases() + words
    return " ".join(n_grams)

def create_BOW_feature_for_english_descriptions(df, input_column: str, output_column: str):
    """ generates "bag of words" aka a list of n-grams between 1 and 3 characters long for each row in the input column
        
        Uses the function "calculate_ngrams_RAKE"
    """

    df["english_BOW"] = ""
    for index, row in df.iterrows():
        if row["description_language"] == "English":
            BOW = calculate_ngrams_RAKE(row[input_column])
        else:
            BOW = ''
        df.at[index,output_column] = BOW

def add_CV_PCA_to_df(df, BOW_column: str, n_components_val: int):
    """
    This function expects a dataframe that has a bag of words column, the name of the bad of words column, 
    and the number of principal components you want to calculate. The output is the original dataframe with the principal component values added as columns.
    """
    
    #instantiating and generating the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(df[BOW_column])

    # convert the count matrix to a dense matrix
    dense_count_matrix = count_matrix.toarray()

    # Scale the data
    scale = StandardScaler()
    scaled_data = scale.fit_transform(dense_count_matrix)

    # Determine the column names for our dense matrix and create a dataframe with the vocabulary as columns:
    temp_dict = {}
    for counter, i in enumerate(list(count.vocabulary_.items())):
            temp_dict[i[1]] = i[0]
    
    column_names = []
    for i in range(len(temp_dict)):
        column_names.append(temp_dict[i])

    # Convert the array back into a dataframe:
    scaled_dataframe=pd.DataFrame(scaled_data, columns= column_names) 

    # PCA analysis
    pca_model = PCA(n_components=n_components_val)
    PCA_components = pca_model.fit_transform(scaled_dataframe)

    # Now append the principal components to the starting df as new features:
    for i in range(PCA_components.shape[1]):
        location= df.shape[1]
        df.insert(location, f"PC{i+1}", PCA_components[:,i].tolist())
        
    return pca_model


def add_TFIDF_PCA_to_df(df, BOW_column: str, n_components_val: int):
    """
    This function expects a dataframe that has a bag of words column, the name of the bad of words column, 
    and the number of principal components you want to calculate. The output is the original dataframe with the principal 
    component values added as columns.
    """
    
    #instantiating and generating the count matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[BOW_column])

    # convert the count matrix to a dense matrix
    dense_X = X.toarray()

    # Scale the data
    scale = StandardScaler()
    scaled_data = scale.fit_transform(dense_X)

    # Determine the column names for our dense matrix and create a dataframe with the vocabulary as columns:
    temp_dict = {}
    for counter, i in enumerate(list(vectorizer.vocabulary_.items())):
            temp_dict[i[1]] = i[0]
    
    column_names = []
    for i in range(len(temp_dict)):
        column_names.append(temp_dict[i])

    # Convert the array back into a dataframe:
    scaled_dataframe=pd.DataFrame(scaled_data, columns= column_names) 

    # PCA analysis
    pca_model = PCA(n_components=n_components_val)
    PCA_components = pca_model.fit_transform(scaled_dataframe)

    # Now append the principal components to the starting df as new features:
    for i in range(PCA_components.shape[1]):
        location= df.shape[1]
        df.insert(location, f"PC{i+1}", PCA_components[:,i].tolist())
        
    return pca_model
