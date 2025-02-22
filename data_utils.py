### Text-to-Vector ###

stop_words = {'i', 'a', 'about', 'an', 'are', 'as', 'at', 'be', 'by', 'com', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with', 'the'} # Basic stop words

def build_vocab(texts):
    vocab = set()

    for text in texts:
        words = text.lower().split() # Splitting the text into words and converting to lowercase for consistency

        # Filtering out stop words
        filtered_words = [word for word in words if word not in stop_words]
        vocab.update(filtered_words)

    return sorted(list(vocab)) # Sorting for consistency 

def text_to_binary_vectors(texts, vocab):
    vectors = []

    for text in texts:
        words = set(text.lower().split()) # Each unique word in the text

        vector = [1 if word in words else 0 for word in vocab]
        vectors.append(vector)
    return vectors


