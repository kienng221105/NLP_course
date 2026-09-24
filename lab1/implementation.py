import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


D = ["cat eats fish",
      "dog eats fish",
      "cat likes fish"]
df2 = pd.DataFrame(D, columns=['text'])
print(df2)

def build_vocabulary(D):
    vocabulary = set()
    for doc in D:
        token = doc.split()
        vocabulary.update(token)
    return sorted(vocabulary)


def compute_counts(D, vocabulary):
    vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros((len(D), len(vocabulary)), dtype=int)
    for i, doc in enumerate(D):
        for token in doc.split():
            j = vocabulary_index.get(token)
            if j is not None:
                matrix[i, j] += 1
    return matrix


def compute_tf(count_matrix):
    count_matrix = np.array(count_matrix, dtype=float, copy=True)
    for i in count_matrix:
        total_count = np.sum(i)
        if total_count > 0:
            i /= total_count
    return count_matrix


def compute_idf(count_matrix):
    count_matrix = np.asarray(count_matrix)
    if count_matrix.ndim != 2 or count_matrix.shape[0] == 0:
        raise ValueError("count_matrix must be a non-empty 2D matrix")

    document_frequency = np.count_nonzero(count_matrix, axis=0)
    return np.log(count_matrix.shape[0] / document_frequency)


def compute_tfidf(tf_matrix, idf_vecto):
    return tf_matrix * idf_vecto


def cosine_similarity(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        return 0.0
    return np.dot(v1, v2) / denominator


vocabulary = build_vocabulary(D)
print(vocabulary)

count_matrix = compute_counts(D, build_vocabulary(D))
print(count_matrix)

print(count_matrix.shape[0])

tf_matrix = compute_tf(count_matrix)
print(tf_matrix)

idf_vecto = compute_idf(count_matrix)
print(idf_vecto)

tfidf_matrix = compute_tfidf(tf_matrix, idf_vecto)
print(tfidf_matrix)

query_vecto = [1, 1, 1, 0, 0]
print(cosine_similarity(query_vecto, tfidf_matrix[0]))
print(cosine_similarity(query_vecto, tfidf_matrix[1]))
print(cosine_similarity(query_vecto, tfidf_matrix[2]))


vocabulary_custom = build_vocabulary(D)
vocabulary_index = {term: i for i, term in enumerate(vocabulary_custom)}

library_vectorizer = CountVectorizer(
    vocabulary=vocabulary_index,
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False
)

X_count_library = library_vectorizer.fit_transform(D)

row_totals = np.asarray(X_count_library.sum(axis=1)).ravel()
X_tf_library = X_count_library.multiply(1 / row_totals[:, None]).toarray()

idf_library = TfidfTransformer(smooth_idf=False, norm=None).fit(X_count_library).idf_ - 1
X_tfidf_library = X_tf_library * idf_library

print('Library vocabulary:', library_vectorizer.get_feature_names_out())
print('Library count matrix:\n', X_count_library.toarray())
print('Library TF matrix:\n', X_tf_library)
print('Library IDF:', idf_library)
print('Library TF-IDF matrix:\n', X_tfidf_library)

print('Counts giống nhau:', np.array_equal(count_matrix, X_count_library.toarray()))
print('TF giống nhau:', np.allclose(tf_matrix, X_tf_library))
print('IDF giống nhau:', np.allclose(idf_vecto, idf_library))
print('TF-IDF giống nhau:', np.allclose(tfidf_matrix, X_tfidf_library))

query_array = np.asarray(query_vecto, dtype=float).reshape(1, -1)
custom_scores = np.array([
    cosine_similarity(query_array.ravel(), row)
    for row in tfidf_matrix
])
library_scores = sklearn_cosine_similarity(query_array, X_tfidf_library).ravel()

print('Custom cosine:', custom_scores)
print('Library cosine:', library_scores)
print('Cosine giống nhau:', np.allclose(custom_scores, library_scores))


assert build_vocabulary(D) == ['cat', 'dog', 'eats', 'fish', 'likes']
assert np.array_equal(
    compute_counts(D, vocabulary_custom),
    np.array([[1, 0, 1, 1, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]])
)
assert np.allclose(compute_tf(count_matrix), tf_matrix)
assert np.allclose(
    compute_idf(count_matrix),
    np.log(np.array([3 / 2, 3, 3 / 2, 1, 3]))
)
assert np.allclose(compute_tfidf(tf_matrix, idf_vecto), tfidf_matrix)
assert np.isclose(cosine_similarity(query_vecto, tfidf_matrix[0]), custom_scores[0])
print('Các kiểm thử cơ bản của Part E đều đạt.')

"""             text
0   cat eats fish
1   dog eats fish
2  cat likes fish
['cat', 'dog', 'eats', 'fish', 'likes']
[[1 0 1 1 0]
 [0 1 1 1 0]
 [1 0 0 1 1]]
3
[[0.33333333 0.         0.33333333 0.33333333 0.        ]
 [0.         0.33333333 0.33333333 0.33333333 0.        ]
 [0.33333333 0.         0.         0.33333333 0.33333333]]
[0.40546511 1.09861229 0.40546511 0.         1.09861229]
[[0.13515504 0.         0.13515504 0.         0.        ]
 [0.         0.3662041  0.13515504 0.         0.        ]
 [0.13515504 0.         0.         0.         0.3662041 ]]
0.8164965809277261
0.7415411516746148
0.1999026538626481
Library vocabulary: ['cat' 'dog' 'eats' 'fish' 'likes']
Library count matrix:
 [[1 0 1 1 0]
 [0 1 1 1 0]
 [1 0 0 1 1]]
Library TF matrix:
 [[0.33333333 0.         0.33333333 0.33333333 0.        ]
 [0.         0.33333333 0.33333333 0.33333333 0.        ]
 [0.33333333 0.         0.         0.33333333 0.33333333]]
Library IDF: [0.40546511 1.09861229 0.40546511 0.         1.09861229]
Library TF-IDF matrix:
 [[0.13515504 0.         0.13515504 0.         0.        ]
 [0.         0.3662041  0.13515504 0.         0.        ]
 [0.13515504 0.         0.         0.         0.3662041 ]]
Counts giống nhau: True
TF giống nhau: True
IDF giống nhau: True
TF-IDF giống nhau: True
Custom cosine: [0.81649658 0.74154115 0.19990265]
Library cosine: [0.81649658 0.74154115 0.19990265]
Cosine giống nhau: True
Các kiểm thử cơ bản của Part E đều đạt."""
