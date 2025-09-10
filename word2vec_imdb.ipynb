# Install required libraries
!pip install gensim
!pip install python-Levenshtein

import gensim
import pandas as pd

# Load dataset
df = pd.read_csv("/content/IMDB Dataset.csv")
print(df.shape)   # (50000, 2)

# Preview one review
print(df.review[0])

# Preprocess reviews
sample_tokens = gensim.utils.simple_preprocess(df.review[0])
print(sample_tokens)

reviews = df.review.apply(gensim.utils.simple_preprocess)

# Initialize Word2Vec model
model = gensim.models.Word2Vec(
    window=10,       # context window
    min_count=2,     # ignore rare words
    workers=4        # parallel processing
)

# Build vocab
model.build_vocab(reviews, progress_per=1000)

# Train model
model.train(reviews, total_examples=model.corpus_count, epochs=model.epochs)

# Save model
model.save("word2vec-imdb.model")

# Query similar words
print(model.wv.most_similar("bad"))
print(model.wv.most_similar("fantasy"))
print(model.wv.similarity(w1="bottle", w2="lion"))
