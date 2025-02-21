from naive_bayes import NaiveBayes

def main():
    nb = NaiveBayes()
    
    # Training data
    x_training = [[1, 0], [0, 1], [1, 1]]
    y_training = [0, 1, 0]


    # Test Data
    X_test = [[1, 0], [0, 1]]
    nb.fit(x_training, y_training)
    predictions = nb.predict(X_test)

    print("Priors:", nb.priors)
    print("Likelihoods:", nb.likelihoods)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()