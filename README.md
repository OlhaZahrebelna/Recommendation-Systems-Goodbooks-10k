# Recommendation-Systems-Goodbooks-10k

## Overview

This project implements and compares several recommendation approaches on the Goodbooks-10k dataset using PyTorch.

The notebook contains four main parts:

1. **Vector Space Model (VSM)** – content-based recommendations using book genres.
2. **Two-Tower Retrieval Model** – neural retrieval based on user and item embeddings.
3. **Neural Collaborative Filtering (NCF)** – concat-based ranking model with an MLP.
4. **Two-Stage Recommendation Pipeline** – combines fast retrieval with neural reranking.

---

## Dataset

The experiments are based on the **Goodbooks-10k** dataset.

Book representations are built from genre/tag information, while user preferences are derived from positive interactions (ratings above the selected threshold).

---

## Project Structure

### Task 1 – Vector Space Model

- Builds normalized item embeddings from genre features.
- Represents each user as the weighted average of liked book embeddings.
- Uses cosine similarity for recommendation.
- Evaluates the model using **Recall@10**.

---

### Task 2 – Two-Tower Retrieval

Implements a retrieval model with:

- User embedding tower
- Item feature tower
- L2-normalized embeddings
- Dot-product similarity
- Binary cross-entropy loss
- Negative sampling

The model is trained to retrieve relevant candidate books efficiently.

---

### Task 3 – Neural Collaborative Filtering (NCF)

Implements a ranking model using:

- Learnable user embeddings
- Item representation network
- Concatenation of user and item vectors
- Multi-layer perceptron (MLP)
- BCEWithLogitsLoss for binary classification

The model predicts the relevance score of user-item pairs.

---

### Task 4 – Two-Stage Recommendation Pipeline

The final recommendation pipeline combines both neural models:

1. **Retrieval stage (Two-Tower)** selects a small set of candidate books.
2. **Ranking stage (NCF)** reranks these candidates to produce the final recommendations.

This architecture provides an effective balance between computational efficiency and recommendation quality.

---

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn

---

## Evaluation

The primary evaluation metric is:

- **Recall@10**
