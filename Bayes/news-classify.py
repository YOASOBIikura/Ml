from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
# cv = CountVectorizer()
# cv_fit = cv.fit_transform(texts)
#
# print(cv.get_feature_names())
# print(cv_fit.toarray())

text = ["The quick brown fox jumped over the lazy dog.", "The dog.", "The fox"]

vectorizer = TfidfVectorizer()
vectorizer.fit(text)

print(vectorizer.vocabulary_)
print(vectorizer.idf_)
