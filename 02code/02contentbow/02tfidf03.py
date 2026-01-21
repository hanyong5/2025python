import pandas as pd
from ast import literal_eval
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances


def load_data(metadata_path="data/movies_metadata.csv", 
              links_path="data/links_small.csv", 
              keywords_path="data/keywords.csv"):
    """
    영화 데이터를 로드하고 병합하는 함수
    
    Args:
        metadata_path: 영화 메타데이터 CSV 파일 경로
        links_path: 링크 데이터 CSV 파일 경로
        keywords_path: 키워드 데이터 CSV 파일 경로
    
    Returns:
        병합된 영화 데이터프레임
    """
    movies_metadata = pd.read_csv(metadata_path)
    links_small = pd.read_csv(links_path)
    movies_keywords = pd.read_csv(keywords_path)
    
    # links_small과 매칭되는 영화만 필터링
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    movies_metadata = movies_metadata[movies_metadata['id'].isin(links_small.astype('str'))]
    
    # 필요한 컬럼만 선택
    movies = movies_metadata[['id','title','genres','popularity','release_date']]
    
    # 키워드 데이터 병합
    movies_keywords['id'] = movies_keywords['id'].astype('str')
    movies = movies.merge(movies_keywords, on=['id'])
    
    return movies


def preprocess_data(movies):
    """
    영화 데이터를 전처리하는 함수
    - 장르와 키워드를 파싱하고 결합
    - 날짜 정보 추출
    - 인기도 로그 변환
    
    Args:
        movies: 전처리할 영화 데이터프레임
    
    Returns:
        전처리된 영화 데이터프레임
    """
    # 장르 파싱 및 정렬
    movies['genres'] = movies['genres'].fillna('[]') \
                        .apply(literal_eval) \
                        .apply(lambda x: sorted(i['name'] for i in x) if isinstance(x, list) else [])
    
    # 키워드 파싱 및 정렬
    movies['keywords'] = movies['keywords'].fillna('[]') \
                        .apply(literal_eval) \
                        .apply(lambda x: sorted(i['name'] for i in x) if isinstance(x, list) else [])
    
    # 장르와 키워드 결합
    movies['str_genres_keywords'] = movies['genres'] + movies['keywords']
    
    # 중복 제거 및 문자열로 변환
    movies['str_genres_keywords'] = movies['str_genres_keywords'] \
                                .apply(lambda x: sorted(list(x))) \
                                .apply(lambda x: " ".join(x) if len(x) > 0 else None)
    
    # 날짜 정보 추출
    movies['release_date'] = pd.to_datetime(movies['release_date'])
    movies['year'] = movies['release_date'].dt.year
    
    # 인기도 로그 변환
    movies['popularity'] = movies['popularity'].astype(float)
    movies['popularity_log'] = np.log(movies['popularity'])
    
    # 결측치 제거
    movies = movies.dropna().reset_index(drop=True)
    
    return movies


def calculate_tfidf_similarity(movies):
    """
    TF-IDF 벡터화를 수행하고 유클리드 거리 기반 유사도를 계산하는 함수
    
    Args:
        movies: 전처리된 영화 데이터프레임
    
    Returns:
        sorted_similarity_of_euclidean: 정렬된 유사도 인덱스 배열
        tfidf_vectorizer: 학습된 TF-IDF 벡터라이저
    """
    # TF-IDF 기반 Contents Based Filtering
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_mat = tfidf_vectorizer.fit_transform(movies['str_genres_keywords'])
    arr_tfidf = tfidf_mat.toarray()
    
    # Euclidean Distances(유클리드 거리)
    similarity_of_euclidean = euclidean_distances(arr_tfidf, arr_tfidf)
    
    # sorted Euclidean Distances
    sorted_similarity_of_euclidean = similarity_of_euclidean.argsort()
    
    return sorted_similarity_of_euclidean, tfidf_vectorizer


def recomm_of_euclidean(movies, sorted_similarity_of_euclidean, title_name, top_k=30):
    """
    유클리드 거리 기반으로 영화를 추천하는 함수
    
    Args:
        movies: 영화 데이터프레임
        sorted_similarity_of_euclidean: 정렬된 유사도 인덱스 배열
        title_name: 추천 기준이 될 영화 제목
        top_k: 추천할 영화 개수 (기본값: 30)
    
    Returns:
        추천된 영화 데이터프레임 (상위 10개)
    """
    movie_of_title = movies[movies['title'] == title_name]
    
    if movie_of_title.empty:
        print(f"'{title_name}' 제목의 영화를 찾을 수 없습니다.")
        return None
    
    print(f'{title_name} 의 장르 : {movie_of_title["str_genres_keywords"].values[0]}')
    
    movie_index_of_title = movie_of_title.index.values[0]
    print(f'\n index : {movie_index_of_title}')
    
    # 유사한 영화 인덱스 추출
    similar_indexes = sorted_similarity_of_euclidean[movie_index_of_title, :top_k*2]
    similar_indexes = similar_indexes.reshape(-1)
    similar_indexes = similar_indexes[similar_indexes != movie_index_of_title]
    print(similar_indexes)
    
    # 인기도와 연도 기준으로 정렬하여 상위 10개 반환
    return movies.iloc[similar_indexes].sort_values(by=['popularity_log','year'], ascending=False)[:10]


def main():
    """
    메인 실행 함수
    데이터 로딩, 전처리, 유사도 계산, 추천을 순차적으로 수행
    """
    # 데이터 로딩
    movies = load_data()
    
    # 데이터 전처리
    movies = preprocess_data(movies)
    
    # TF-IDF 유사도 계산
    sorted_similarity_of_euclidean, tfidf_vectorizer = calculate_tfidf_similarity(movies)
    
    # 영화 추천
    recomm_movies = recomm_of_euclidean(movies, sorted_similarity_of_euclidean, 'Jumanji')
    
    if recomm_movies is not None:
        print(recomm_movies[['title','popularity_log','year']])


if __name__ == "__main__":
    main()










