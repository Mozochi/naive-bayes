import math

from config import smoothingFactor

class NaiveBayes:
    def __init__(self, smoothing=smoothingFactor):
        self.priors = {}
        self.likelihoods = {}
        self.smoothing = smoothing


    def fit(self, x, y):
        totalDocuments  = len(y)
        uniqueClasses = set(y)

        # Counting how many times class c appears in y
        for classLabel in uniqueClasses: 
            documentsInClass = y.count(classLabel)
            self.priors[classLabel] = documentsInClass / totalDocuments 
        
        numFeatures = len(x[0]) # Number of features per document
        for classLabel in uniqueClasses:
            self.likelihoods[classLabel] = {}

            # All the feature vectors for this class
            classDocuments = [x[i] for i in range(totalDocuments) if y[i] == classLabel]
            # For each feature position
            for featurePosition in range(numFeatures):
                # Counting how many times this feature is 1 for this class
                featurePositiveCount = sum(1 for document in classDocuments if document[featurePosition] == 1)
                # Laplace Smoothing (positive count + smoothing) / (class documents + smoothing * possible values)
                self.likelihoods[classLabel][featurePosition] = (featurePositiveCount + self.smoothing) / (len(classDocuments) + self.smoothing * 2)
        



    def predict(self, x):
        predictions = []
        for document in x: # Classify each document
            scores= {}

            # For each possible class
            for classLabel in self.priors:
                # Start with log of the prior probability
                score = math.log(self.priors[classLabel])
                # Add the log of the likelihood of each feature
                for featurePosition, featureValue in enumerate(document):
                    likelihood = self.likelihoods[classLabel][featurePosition]
                    if featureValue == 1:
                        score += math.log(likelihood)
                    else:
                        score += math.log(1 - likelihood)
                scores[classLabel] = score

            # Pick the class wit the highest score
            predictedClass = max(scores, key=lambda k: scores[k])
            predictions.append(predictedClass)

        return predictions