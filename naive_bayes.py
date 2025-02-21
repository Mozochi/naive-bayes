from config import smoothingFactor

class NaiveBayes:
    def __init__(self, smoothing=smoothingFactor):
        self.priors = {}
        self.likelihoods = {}
        self.smoothing = smoothing


    def fit(self, x, y):
        # Train model by calculating priors and likelihoods
        totalDocuments  = len(y)
        uniqueClasses = set(y)

        # Counting how many times class c appears in y
        for classLabel in uniqueClasses: 
            documentsInClass = y.count(classLabel)
            self.priors[classLabel] = documentsInClass / totalDocuments 
        

        



    def predict(self, x):
        # Return the predicted class labels
        pass