from naive_bayes import NaiveBayes
from data_utils import dummy_data

def main():
    nb = NaiveBayes()
    x_training, y_training, X_test = dummy_data()
    
    # Training data
    nb.fit(x_training, y_training)
    predictions = nb.predict(X_test)

    print("Priors:", nb.priors)
    print("Likelihoods:", nb.likelihoods)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()