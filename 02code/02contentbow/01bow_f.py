import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(metadata_path='data/movies_metadata.csv', links_path='data/links_small.csv'):
    """데이터 로딩"""
    movies_metadata = pd.read_csv(metadata_path)
    links_small = pd.read_csv(links_path)
    return movies_metadata, links_small


def preprocess_data(movies_metadata, links_small):
    """데이터 전처리"""
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    movies_metadata_small = movies_metadata[movies_metadata['id'].isin(links_small.astype('str'))]

    movies = movies_metadata_small[['title', 'genres', 'popularity', 'release_date']].copy()

    movies['str_genres'] = movies['genres']\
                        .apply(literal_eval)\
                        .apply(lambda x: sorted([i['name'] for i in x]) if isinstance(x, list) else [])\
                        .apply(lambda x: " ".join(x) if len(x) > 0 else None)

    movies['release_date'] = pd.to_datetime(movies['release_date'])
    movies['year'] = movies['release_date'].dt.year

    movies = movies.dropna()
    movies = movies.reset_index(drop=True)

    return movies


def compute_similarity(movies):
    """유사도 계산"""
    # CountVectorizer는 문장에 포함된 단어 수를 이용하여 벡터로 변환하는 객체이다.
    bow_vectorizer = CountVectorizer()

    # 장르 데이터를 2차원의 메트릭스 데이터로 변환
    genre_mat = bow_vectorizer.fit_transform(movies['str_genres'])

    # 영화별 유사도 연산
    similarity_of_genre = cosine_similarity(genre_mat, genre_mat)
    sorted_similarity_of_genre = similarity_of_genre.argsort()[:, ::-1]

    return sorted_similarity_of_genre


def recommendation(movies, sorted_similarity_of_genre, title_name, top_k=5):
    """영화 추천"""
    # 기준 영화 추출
    movie_of_title = movies[movies['title'] == title_name]
    print(f"{title_name}의 장르: {movie_of_title['str_genres'].values[0]}")

    # 기준 영화 인덱스
    movie_index_of_title = movie_of_title.index.values[0]

    # 기준 영화를 기준으로 가장 유사도가 높은 영화들의 인덱스 추출
    similar_indexes = sorted_similarity_of_genre[movie_index_of_title, :(top_k*2)]
    # 메트릭스(2차원의 데이터)를 벡터(1차원의 데이터)로 변환
    similar_indexes = similar_indexes.reshape(-1)
    # 기준 영화 인덱스는 제외
    similar_indexes = similar_indexes[similar_indexes != movie_index_of_title]

    # 유사도 기반으로 추출된 영화 추천
    return movies.iloc[similar_indexes].sort_values(by=['year'], ascending=False)[:top_k]


def main():
    # 데이터 로딩
    movies_metadata, links_small = load_data()

    # 데이터 전처리
    movies = preprocess_data(movies_metadata, links_small)

    # 유사도 계산
    sorted_similarity_of_genre = compute_similarity(movies)

    # 영화 추천
    recommendation_movies = recommendation(movies, sorted_similarity_of_genre, 'Toy Story', top_k=5)
    print(recommendation_movies['title'].tolist())


if __name__ == '__main__':
    main()
