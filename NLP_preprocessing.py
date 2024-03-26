from langdetect import detect, DetectorFactory
from iso639 import Lang
from rake_nltk import Rake

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

def create_BOW_feature_for_english_descriptions(df, input_column: str):
    """ generates "bag of words" aka a list of n-grams between 1 and 3 characters long for each row in the input column
        
        Uses the function "calculate_ngrams_RAKE"
    """

    df["english_BOW"] = ""
    for index, row in df.iterrows():
        if row["description_language"] == "English":
            BOW = calculate_ngrams_RAKE(row[input_column])
        else:
            BOW = ['no english description']
        df.at[index,'english_BOW'] = BOW
