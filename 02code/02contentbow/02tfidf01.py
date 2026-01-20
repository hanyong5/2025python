from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


documents = [
    "부산 여행 바다 맛집",
    "부산 해변 바다 산책",
    "서울 맛집 데이트",
    "제주도 여행 자연"
]


vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

# print(vectorizer.vocabulary_)
# print(tfidf.toarray()[0])
# print(tfidf.toarray()[1])
# print(tfidf.toarray()[2])
# print(tfidf.toarray()[3])

df_tfidf = pd.DataFrame(
    tfidf.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[ f'문서 {i}' for i in range(len(documents))]
)

plt.figure(figsize=(12,4))
sns.heatmap(
    df_tfidf,
    cmap="Reds",
    annot=True
)
plt.show()