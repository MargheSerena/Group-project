import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from iso639 import Lang
from rake_nltk import Rake

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

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

def concat_nouns(text):
    """
    concatenate and apply lowercase lettering to proper nouns like names, publishers, and book titles
    """

    # Step 1: remove leading or ending brackets (if applicable) and internal quote marks
    text = text.strip("[]")
    text = text.replace("'", "")

    # Step 2: If there are multiple nouns, split at the comma
    text = text.split(", ")

    # Step 3: Concatenate each noun and put all letters in lowercase:
    text = [x.replace(" ", "").lower() for x in text]

    # Step 4: Convert the list of tokens to a string
    text = ' '.join(text)

    return text

def add_tokens_to_description(df):
    df["description"] += " " + df["authors"].apply(concat_nouns)
    df["description"] += " " + df["publisher"].apply(concat_nouns)
    df["description"] += " " + df["Title"].str.lower()

def calculate_ngrams_RAKE_original(text: str):
    """ generates a list of n-grams for a given input text using RAKE
    """
    r_unigram = Rake()
    r_phrase = Rake(min_length=2, max_length=3)
    r_unigram.extract_keywords_from_text(text)
    r_phrase.extract_keywords_from_text(text)
    
    keyword_dict_scores = r_unigram.get_word_degrees()
    words = list(keyword_dict_scores.keys())
    words
    
    # n_grams = r_phrase.get_ranked_phrases() + words
    # return " ".join(n_grams)
    return " ".join(words)

def calculate_ngrams_RAKE(text: str):
    """ generates a list of n-grams for a given input text using RAKE
    """

    r_unigram = Rake()
    r_unigram.extract_keywords_from_text(text)
    
    keyword_dict_scores = r_unigram.get_word_degrees()
    words = list(keyword_dict_scores.keys())
    
    return " ".join(words)

def create_BOW_feature_for_english_descriptions_STEMS(df, input_column: str, output_column: str):
    """ generates "bag of words" aka a list of unigrams for 
        each row in the input column
        
        Uses the function "calculate_ngrams_RAKE"

        Also breaks tokens down into stems to reduce the size of the vocabulary
    """
    ps = PorterStemmer()        
    
    df["english_BOW"] = ""
    for index, row in df.iterrows():
        if row["description_language"] == "English":
            BOW = calculate_ngrams_RAKE(row[input_column])
        else:
            BOW = ''

        try:
            words = word_tokenize(BOW)
            stems = ""
            for w in words:
                stems += " " + ps.stem(w)
        except:
            pass
            
        df.at[index,output_column] = stems


def create_BOW_feature_for_english_descriptions(df, input_column: str, output_column: str):
    """ generates "bag of words" aka a list of unigrams for 
        each row in the input column
        
        Uses the function "calculate_ngrams_RAKE"

        Also breaks tokens down into stems to reduce the size of the vocabulary
    """
    ps = PorterStemmer()        
    
    df["english_BOW"] = ""
    for index, row in df.iterrows():
        if row["description_language"] == "English":
            BOW = calculate_ngrams_RAKE(row[input_column])
        else:
            BOW = ''
            
        df.at[index,output_column] = BOW

def create_tokens(df, input_column: str, output_column: str):
    """ generates tokens aka a list of unigrams for 
        each row in the input column. Uses the function "calculate_ngrams_RAKE"
    """     
    df[output_column] = df["description"].apply(calculate_ngrams_RAKE)



def add_TFIDF_iPCA_to_df(df, BOW_column: str, n_components_val: int):
    """
    This function expects a dataframe that has a bag of words column, the name of the bad of 
    words column, 
    and the number of principal components you want to calculate. The output is the original 
    dataframe with the principal 
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

    # Determine the column names for our dense matrix and create a dataframe with the 
    # vocabulary as columns:
    temp_dict = {}
    for counter, i in enumerate(list(vectorizer.vocabulary_.items())):
            temp_dict[i[1]] = i[0]
    
    column_names = []
    for i in range(len(temp_dict)):
        column_names.append(temp_dict[i])

    # Convert the array back into a dataframe:
    scaled_dataframe=pd.DataFrame(scaled_data, columns= column_names) 

    # PCA analysis
    ipca_model = IncrementalPCA(n_components=n_components_val)
    PCA_components = ipca_model.fit_transform(scaled_dataframe)

    # Now append the principal components to the starting df as new features:
    for i in range(PCA_components.shape[1]):
        location= df.shape[1]
        df.insert(location, f"PC{i+1}", PCA_components[:,i].tolist())
        
    return ipca_model

def add_TFIDF_tSVD_to_df(df, BOW_column: str, n_components_val: int, train_index, test_index):
    """
    This function expects a dataframe that has a bag of words column, the name of the bag 
    of words column, and the number of tSVD components you want to calculate. The 
    output includes the following variables: tSVD_model, tSVD_components_X_train, tSVD_components_X_test, X_train_df, X_test_df

    The train and test dataframes are returned will have the tSVD features added.
    """
    # Split the incoming dataframe into train and test slices base on the list of train and test indices provided:
    X_train_df = df[df["index"].isin(train_index)]
    X_test_df = df[df["index"].isin(test_index)]
    
    #instantiating and generating the tfidf
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_df[BOW_column])
    X_test_tfidf = vectorizer.transform(X_test_df[BOW_column])

    # convert the tfidf matrix to a dense matrix
    dense_X_train_tfidf = X_train_tfidf.toarray()
    dense_X_test_tfidf = X_test_tfidf.toarray()

    # # combine the arrays:
    # dense_X = np.concatenate((dense_X_train_tfidf, dense_X_test_tfidf), axis = 0)

    # Determine the column names for our dense matrix and create a dataframe with the 
    # vocabulary as columns:
    temp_dict = {}
    for counter, i in enumerate(list(vectorizer.vocabulary_.items())):
            temp_dict[i[1]] = i[0]
    
    column_names = []
    for i in range(len(temp_dict)):
        column_names.append(temp_dict[i])

    # Convert the array back into a dataframe:
    scaled_dataframe_X_train=pd.DataFrame(dense_X_train_tfidf, columns= column_names)
    scaled_dataframe_X_test=pd.DataFrame(dense_X_test_tfidf, columns= column_names) 

    
    # tSVD analysis
    tSVD_model = TruncatedSVD(n_components=n_components_val)
    tSVD_components_X_train = tSVD_model.fit_transform(scaled_dataframe_X_train)
    tSVD_components_X_test = tSVD_model.transform(scaled_dataframe_X_test)

    # # Now append the tSVD components to the train and test dataframes as new features:
    for i in range(tSVD_components_X_train.shape[1]):
        location= X_train_df.shape[1]
        X_train_df.insert(location, f"tSVD{i+1}", tSVD_components_X_train[:,i].tolist())

    for i in range(tSVD_components_X_test.shape[1]):
        location= X_test_df.shape[1]
        X_test_df.insert(location, f"tSVD{i+1}", tSVD_components_X_test[:,i].tolist())
    
        
    return tSVD_model, tSVD_components_X_train, tSVD_components_X_test, X_train_df, X_test_df