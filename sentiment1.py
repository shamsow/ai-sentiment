import os
import nltk
from nltk.tokenize import MWETokenizer
from nltk.stem.snowball import EnglishStemmer

# tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
mwe_tknzr = MWETokenizer(separator=' ') # Multi-word-expression tokenizer
stemmer = EnglishStemmer() # Word stemmer
scores = {} # dict of words and their scores

def main():
    load_data() # populate the scores dict
    s = input("sentence: ")
    print(get_sentiment(s))


def check_mwe(word):
    """If there are multiple words in the line, add the multi-word-expression to the tokenizer"""
    tokens = word.split()
    if len(tokens) != 1:
        mwe_tknzr.add_mwe(tokens)


def load_data(directory='corpus2'):
    """Extract words and their scores from the desired corpus"""
    # Read data in from files
    with open(os.path.join(directory, 'positives.txt')) as f:
        for line in f.read().splitlines():
            
            row = line.rstrip('\n').split(',')
            word = row[0]
            check_mwe(word)
            intensity = float(row[-1])
            
            scores[word] = intensity

    with open(os.path.join(directory, 'negatives.txt')) as f:
        for line in f.read().splitlines():

            row = line.rstrip('\n').split(',')
            word = row[0]
            check_mwe(word)
            intensity = float(row[-1])
            
            scores[word] = intensity


def extract_words(sentence):
    """Convert a sentence into a set of tokens taking in to account multi-word-expressions"""
    # Create simple tokens of all words in the sentence
    words = [word.lower() for word in nltk.word_tokenize(sentence) if any(c.isalpha() for c in word)]
    # Split the tokens into multi-word-expressions, if any
    tokens = mwe_tknzr.tokenize(words)
    # print(tokens)
    return set(tokens)


def compute_score(sentence):
    """Calculate the sentiment score for the given sentence"""
    document_words = extract_words(sentence)
    score = 0
    for word in document_words:
        grade = scores.get(word.lower(), 0)
        if grade == 0:
            # If the word isn't in the scores dict, try to get the stemmed version of the word from the dict (cars becomes car, abandoned becomes abandon, etc.)
            grade = scores.get(stemmer.stem(word.lower()), 0)
        score += grade
    # Convert the score in to a -1 to 1 scale
    score = score / len(document_words)
    # print(score)
    return score


def get_sentiment(sentence):
    """Classify the sentence to be positive or negative"""
    score = compute_score(sentence)
    if score > 0:
        return ("Positive", score)
    else:
        return ("Negative", score)


if __name__ == "__main__":
    main()