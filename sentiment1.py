import os
import nltk
from nltk.tokenize import MWETokenizer
from nltk.stem.snowball import EnglishStemmer

# tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

def main():
    classifier = Sentiment('corpus2')
    classifier.start()

class Sentiment:
    mwe_tknzr = MWETokenizer(separator=' ') # Multi-word-expression tokenizer
    stemmer = EnglishStemmer() # Word stemmer
    scores = {} # dict of words and their scores

    def __init__(self, corpus):
        self.load_data(corpus)

    
    def check_mwe(self, word):
        """If there are multiple words in the line, add the multi-word-expression to the tokenizer"""
        tokens = word.split()
        if len(tokens) != 1:
            self.mwe_tknzr.add_mwe(tokens)


    def load_data(self, directory):
        """Extract words and their scores from the desired corpus"""
        # Read data in from files
        with open(os.path.join(directory, 'positives.txt')) as f:
            for line in f.read().splitlines():
                
                row = line.rstrip('\n').split(',')
                word = row[0]
                self.check_mwe(word)
                intensity = float(row[-1])
                
                self.scores[word] = intensity

        with open(os.path.join(directory, 'negatives.txt')) as f:
            for line in f.read().splitlines():

                row = line.rstrip('\n').split(',')
                word = row[0]
                self.check_mwe(word)
                intensity = float(row[-1])
                
                self.scores[word] = intensity


    def extract_words(self, sentence):
        """Convert a sentence into a set of tokens taking in to account multi-word-expressions"""
        # Create simple tokens of all words in the sentence
        words = [word.lower() for word in nltk.word_tokenize(sentence) if any(c.isalpha() for c in word)]
        # Split the tokens into multi-word-expressions, if any
        tokens = self.mwe_tknzr.tokenize(words)
        # print(tokens)
        return set(tokens)


    def compute_score(self, sentence):
        """Calculate the sentiment score for the given sentence"""
        document_words = self.extract_words(sentence)
        score = 0
        for word in document_words:
            grade = self.scores.get(word.lower(), 0)
            if grade == 0:
                # If the word isn't in the scores dict, try to get the stemmed version of the word from the dict (cars becomes car, abandoned becomes abandon, etc.)
                grade = self.scores.get(self.stemmer.stem(word.lower()), 0)
            score += grade
        # Convert the score in to a -1 to 1 scale
        score = score / len(document_words)
        # print(score)
        return score


    def get_sentiment(self, sentence):
        """Classify the sentence to be positive or negative"""
        score = self.compute_score(sentence)
        if score > 0:
            return ("Positive", score)
        else:
            return ("Negative", score)
    
    def start(self):
        print("Keep entering sentences to get a sentiment estimation from the AI")
        print("Type 'exit' to quit")
        while True:
            s = input("Sentence: ")
            if s == 'exit':
                break
            print(self.get_sentiment(s))


if __name__ == "__main__":
    main()