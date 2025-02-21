from config import smoothingFactor

class NaiveBayes:
    def __init__(self, smoothing=smoothingFactor):
        self.priors = {}
        self.likelihoods = {}
        self.smoothing = smoothing


    def fit(self, x, y):
        # Train model by calculating priors and likelihoods
        pass


    def predict(self, x):
        # Return the predicted class labels
        pass