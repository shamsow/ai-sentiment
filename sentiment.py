# This one was largely unsuccesful
# I'm guessing the problem lies in me trying to use just positive or negative words to train an ai to classify sentences
# Refer to sentiment1.py for working version


# Train NaiveBayes classifier from text files
import nltk
import os
import sys

# from senticnet4 import senticnet

# print(senticnet['a_little'][7])

def main():

    # Read data from files
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py corpus")
    positives, negatives = load_data(sys.argv[1])
    # print(positives)
    # Create a set of all words
    words = set()
    
    # for document in positives:
    #     words.update(document)
    # for document in negatives:
    #     words.update(document)

    words.update(positives[0].keys())
    words.update(negatives[0].keys())
    # print(words)

    # Extract features from text
    training = []
    training.extend(generate_features(positives[0], words, "Positive"))
    training.extend(generate_features(negatives[0], words, "Negative"))
    # print(training)
    # Classify a new sample
    classifier = nltk.NaiveBayesClassifier.train(training)
    s = input("sentence: ")
    result = (classify(classifier, s, words, positives[0], negatives[0]))
    for key in result.samples():
        print(f"{key}: {result.prob(key):.4f}")

    classifier.show_most_informative_features(5)


def extract_words(document):
    return set(
        word.lower() for word in nltk.word_tokenize(document)
        if any(c.isalpha() for c in word)
    )


def load_data(directory):
    result = []
    for filename in ["positives.txt", "negatives.txt"]:

        with open(os.path.join(directory, filename)) as f:
            # result.append([
            #     extract_words(line)
            #     for line in f.read().splitlines()
            # ])
            pol = []
            scores = {}
            for line in f.read().splitlines():
                
                word, intensity = line.rstrip('\n').split(',')
                scores[word] = float(intensity)
                pol.append(scores)
            result.append(pol)
            #  result.append([
            #     {word} for word in f.read().splitlines()
            # ])
    # print(result[0][:10])
    return result


def generate_features(documents, words, label):

    # make set of highest intensity words and add a feature that checks words against these intense words and assigns a boolean
    features = []
# for document in documents:
    # print(document)


    # features.append(({
    #     word : documents[word]
    #     for word in words
    # }, label))
    for word in documents:
        features.append(({'score': documents[word]}, label))

    print(features[:20])
    return features


def classify(classifier, document, words, positives, negatives):
    document_words = extract_words(document)
    score = 0
    for word in document_words:
        if word in positives:
            score += positives.get(word, 0)
        elif word in negatives:
            score += negatives.get(word, 0)
    
    score = score / len(document_words)
    print(score)
    # features = {
    #     'score': (word in document_words)
    #     for word in words
    # }
    features = {
        'score': score
    }

    return classifier.prob_classify(features)


if __name__ == "__main__":
    main()


# split sentence into n grams using nltk.ngrams

# Get trending topics from Twitter api
# Using twint search to get tweets
# Use nltk NaiveBayes classifier to classify each tweet
# Return the overall sentiment of the trend