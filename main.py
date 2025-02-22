from naive_bayes import NaiveBayes
from data_utils import build_vocab, text_to_binary_vectors

def main():
    # Training Data
    trainTexts = [
        "I love pizza",                 # Class 1 (Positive)
        "I enjoy coding",               # Class 1 (Positive)
        "I hate spam notifications",    # Class 0 (Negative)
        "I dislike noisy alerts",       # Class 0 (Negative)
        "I hate rainy days",            # Class 0 (Negative)
        "I love machine learning",      # Class 1 (Positive)
        "I hate sitting in traffic"     # Class 0 (Negative)
        ]
    y_train = [1, 1, 0, 0, 0, 1, 0]

    # Building the vocab from the training data
    vocabulary = build_vocab(trainTexts)
    print("Vocabulary:", vocabulary)
  
    # Converting the texts to binary vectors
    X_train = text_to_binary_vectors(trainTexts, vocabulary)
    print("Training vectors:", X_train)

    # Training the model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    print("Priors:", nb.priors)
    print("Likelihoods:", nb.likelihoods)

    # Testing Data
    testTexts = ["I love coding", "I hate pizza",]

    X_test = text_to_binary_vectors(testTexts, vocabulary)
    predictions = nb.predict(X_test)
    print("Test vectors", X_test)
    print("Predictions:", predictions)



if __name__ == "__main__":
    main()