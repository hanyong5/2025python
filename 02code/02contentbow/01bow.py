import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

tqdm.pandas()


print("데이터 로드중...")
movies_metadata = pd.read_csv("data/movies_metadata.csv")
links_small = pd.read_csv("data/links_small.csv")
print("데이터 로드완료...")
print("-"*50)

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
# (9099, 24)
movies_metadata_small= movies_metadata[movies_metadata['id'].isin(links_small.astype('str'))]

movies = movies_metadata_small[['title','popularity','genres','release_date']].copy()
movies['str_genres'] = movies['genres'] \
    .progress_apply(literal_eval) \
    .progress_apply(lambda x: sorted(i['name'] for i in x) if isinstance(x, list) else [] ) \
    .progress_apply(lambda x:" ".join(x) if len(x)>0 else None )


movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['year'] = movies['release_date'].dt.year

movies = movies.dropna()
movies = movies.reset_index(drop=True)

# BoW / contents base filtering
bow_vector = CountVectorizer()
genre_mat = bow_vector.fit_transform(movies['str_genres'])

# 코사이유사도 측정 (9064, 9064)
print("유사도 계산중...")
similarity_of_genre = cosine_similarity(genre_mat,genre_mat)
print("유사도 계산완료...")
print("-"*50)
sorted_similarity_of_genre = similarity_of_genre.argsort()
sorted_similarity_of_genre = sorted_similarity_of_genre[:,::-1]


def recommend(title_name, top_k=5):
    movies_of_title= movies[movies['title'] == title_name]
    print(f'{title_name}의 장르 : {movies_of_title['str_genres'].values[0]}')

    movies_index_of_title = movies_of_title.index.values[0]
    similar_indexes = sorted_similarity_of_genre[movies_index_of_title,:(top_k*2)]

    similar_indexes = similar_indexes.reshape(-1)
    similar_indexes = similar_indexes[similar_indexes != movies_index_of_title]

    print(similar_indexes)
    return movies.iloc[similar_indexes].sort_values(by=['year'],ascending=False)[:top_k]
    

result_movie = recommend('Jumanji', top_k=5)
print(result_movie['title'].tolist())