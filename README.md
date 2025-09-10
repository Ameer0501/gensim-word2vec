# Word2Vec Implementation for Text Similarity
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-green.svg)](https://radimrehurek.com/gensim/)
[![Stars](https://img.shields.io/github/stars/Ameer0501/gensim-word2vec.svg?style=social)](https://github.com/Ameer0501/gensim-word2vec)
# 🎬 Word2Vec on IMDB Movie Reviews

This project demonstrates how to train **Word2Vec embeddings** using the **IMDB Movie Reviews dataset**.  
We explore the limitations of Bag of Words (BoW), introduce Word2Vec, and show how to train embeddings with **Gensim**.

---

## 🔹 Introduction: Word Embeddings
Word embeddings are **dense vector representations of words** that capture semantic meaning.  
Example:  
- `"good"` is close to `"great"`  
- `"bad"` is close to `"awful"`  

Unlike one-hot vectors, embeddings capture relationships between words.

---

## 🔹 Bag of Words (BoW)
BoW represents sentences by word counts.

Example:  
Sentence 1: "The cat sat on the mat"
Sentence 2: "The dog sat on the mat"

BoW vectors:  
- Sentence 1 → [cat:1, dog:0, sat:1, mat:1]  
- Sentence 2 → [cat:0, dog:1, sat:1, mat:1]  

✅ Simple  
❌ Ignores word order  
❌ Large sparse vectors  
❌ No semantic meaning (cat ≠ dog)  

---

## 🔹 Word2Vec
Word2Vec learns **dense embeddings** using neural networks.  
Two architectures:

1. **CBOW (Continuous Bag of Words)**  
   - Predicts the target word given its context.  
   - Example: `the cat ___ on the mat` → predict `"sat"`.  

2. **Skip-gram**  
   - Predicts context words given the target word.  
   - Example: input `"sat"` → predict `"the", "cat", "on", "mat"`.  

---

## 🔹 CBOW vs Skip-gram
- **CBOW**: Context → predict target (faster, works well on large datasets).  
- **Skip-gram**: Target → predict context (slower, better for small datasets & rare words).  

---

## 🔹 Gensim
[Gensim](https://radimrehurek.com/gensim/) is a Python library for topic modeling and embeddings.  
It provides efficient implementations of **Word2Vec, FastText, and Doc2Vec**.

---

## 🔹 Dataset
We use the **IMDB Movie Reviews dataset** (`IMDB Dataset.csv`).  
- 50,000 reviews  
- 2 columns: `review`, `sentiment`

---

## 🔹 Implementation (Jupyter Notebook)
See [word2vec_imdb.py](word2vec_imdb.py) for full code. 
See [requirements.txt](requirements.txt) for dependencies.

### Steps:
1. Install dependencies  
2. Load and preprocess reviews  
3. Train Word2Vec using Gensim  
4. Save model  
5. Query word similarities  

---

## 🔹 Example Results

```python
model.wv.most_similar("bad")
# [('awful', 0.76), ('terrible', 0.75), ('good', 0.73)]

model.wv.most_similar("fantasy")
# [('fairytale', 0.70), ('adventure', 0.67), ('magical', 0.60)]

model.wv.similarity("bottle", "lion")
# 0.1388
```
## 🔹 Key Takeaways

✅ Bag of Words (BoW)

Simple to implement

❌ Lacks semantic meaning

✅ Word2Vec

Captures semantic relationships (e.g., good ≈ great, bad ≈ awful)

⚡ CBOW

Faster, works well on large datasets

🔍 Skip-gram

Slower, but better for rare words & small datasets

🛠 Tools

Gensim
 makes Word2Vec training easy & scalable
