from langdetect import detect, DetectorFactory
from iso639 import Lang



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

def TEST_func():
    print("this is a test")

def other_test():
    print("otehr")
