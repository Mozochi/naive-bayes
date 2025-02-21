from naive_bayes import NaiveBayes

def main():
    nb = NaiveBayes()
    print (f"Naive Bates model has been initialized with smoothing: {nb.smoothing}")

if __name__ == "__main__":
    main()