
from sklearn.feature_extraction.text import TfidfVectorizer

a = {}
a.setdefault("word", 1)
a.setdefault("hua", 2)
for i, item in enumerate(a):
    print(item[0])