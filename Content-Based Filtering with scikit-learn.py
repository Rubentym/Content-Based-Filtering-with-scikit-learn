from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Example dataset (item descriptions)
items = ["Movie: Action thriller with intense plot twists.",
         "TV Show: Romantic comedy set in a small town.",
         "Book: Fantasy adventure with magic and dragons."]

# Calculate TF-IDF vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(items)

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Example usage:
item_idx = 0
sim_scores = list(enumerate(cosine_sim[item_idx]))
print("Similarity scores for item", item_idx, ":", sim_scores)
