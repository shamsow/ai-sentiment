# This one is able to classify sentences
# However, the approach is not even close to perfect
# The fundamental challenge here lies in seperating the sentence in a way that will allow the program to 
# utilise the data in a much more reliable way
# Right now, all the words are being considered one by one, whereas an ideal approach would be to group words 
# to better model the context of the sentences

import random
import os
import sys
import numpy as np

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=3)
model = GaussianNB()




def main():

    # Read data from files
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py corpus")

    data, scores = load_data(sys.argv[1])

    # Separate data into training and testing groups
    # evidence = np.reshape([row["evidence"] for row in data], (-1, 1))
    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]
    # print(evidence)
    # print(labels)
    X_training, X_testing, y_training, y_testing = train_test_split(
        evidence, labels, test_size=0.2
    )

    # Fit model
    model.fit(X_training, y_training)

    # Make predictions on the testing set
    predictions = model.predict(X_testing)

    # Compute how well we performed
    correct = (y_testing == predictions).sum()
    incorrect = (y_testing != predictions).sum()
    total = len(predictions)

    # Print results
    print(f"Results for model {type(model).__name__}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {100 * correct / total:.2f}%")

    query = input("sentence: ")
    score = compute_score(query, scores)
    
    print(classify(model, score))



def compute_score(sentence, scores):
    document_words = sentence.split()

    score = 0
    for word in document_words:
        score += scores.get(word, 0)

    score = score / len(document_words)
    return score

def load_data(directory):
    data = []
    scores = {}

    # Read data in from files
    with open(os.path.join(directory, 'positives.txt')) as f:
        for line in f.read().splitlines():
            
            row = line.rstrip('\n').split(',')

            word = row[0]
            intensity = float(row[-1])
            

            scores[word] = float(intensity)
            # scores[word] = [float(cell) for cell in row[1:]]
            data.append({
                "evidence": [float(cell) for cell in row[1:]],
                "label": "positive"
            })
            
    with open(os.path.join(directory, 'negatives.txt')) as f:
        for line in f.read().splitlines():
            
            # word, intensity = line.rstrip('\n').split(',')
            # scores[word] = float(intensity)
            # data.append({
            #     "evidence": float(intensity),
            #     "label": "negative"
            # })
            row = line.rstrip('\n').split(',')

            word = row[0]
            intensity = float(row[-1])
            

            scores[word] = float(intensity)
            data.append({
                "evidence": [float(cell) for cell in row[1:]],
                "label": "negative"
            })

    return data, scores

def classify(classifier, score):
    score = np.reshape(score, (1, -1))
    return classifier.predict(score)

if __name__ == "__main__":
    main()

# Get trending topics from Twitter api
# Using twint search to get tweets
# Use NaiveBayes classifier to classify each tweet
# Return the overall sentiment of the trend