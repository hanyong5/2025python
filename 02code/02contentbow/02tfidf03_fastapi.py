from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from contextlib import asynccontextmanager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수로 데이터 저장
movies = None
sorted_similarity_of_euclidean = None
tfidf_vectorizer = None


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
    logger.info("데이터 로딩 시작...")
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
    
    logger.info(f"데이터 로딩 완료: {len(movies)}개 영화")
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
    logger.info("데이터 전처리 시작...")
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
    
    logger.info(f"데이터 전처리 완료: {len(movies)}개 영화")
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
    logger.info("TF-IDF 유사도 계산 시작...")
    # TF-IDF 기반 Contents Based Filtering
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_mat = tfidf_vectorizer.fit_transform(movies['str_genres_keywords'])
    arr_tfidf = tfidf_mat.toarray()
    
    # Euclidean Distances(유클리드 거리)
    similarity_of_euclidean = euclidean_distances(arr_tfidf, arr_tfidf)
    
    # sorted Euclidean Distances
    sorted_similarity_of_euclidean = similarity_of_euclidean.argsort()
    
    logger.info("TF-IDF 유사도 계산 완료")
    return sorted_similarity_of_euclidean, tfidf_vectorizer


def recomm_of_euclidean(movies, sorted_similarity_of_euclidean, title_name, top_k=30, num_recommendations=10):
    """
    유클리드 거리 기반으로 영화를 추천하는 함수
    
    Args:
        movies: 영화 데이터프레임
        sorted_similarity_of_euclidean: 정렬된 유사도 인덱스 배열
        title_name: 추천 기준이 될 영화 제목
        top_k: 후보 영화 개수 (기본값: 30)
        num_recommendations: 최종 추천 영화 개수 (기본값: 10)
    
    Returns:
        추천된 영화 데이터프레임, 입력 영화 정보
    """
    movie_of_title = movies[movies['title'] == title_name]
    
    if movie_of_title.empty:
        return None, None
    
    movie_index_of_title = movie_of_title.index.values[0]
    movie_info = movie_of_title.iloc[0]
    
    # 유사한 영화 인덱스 추출
    similar_indexes = sorted_similarity_of_euclidean[movie_index_of_title, :top_k*2]
    similar_indexes = similar_indexes.reshape(-1)
    similar_indexes = similar_indexes[similar_indexes != movie_index_of_title]
    
    # 인기도와 연도 기준으로 정렬하여 상위 N개 반환
    recommended_movies = movies.iloc[similar_indexes].sort_values(
        by=['popularity_log','year'], 
        ascending=False
    )[:num_recommendations]
    
    return recommended_movies, movie_info


# Pydantic 모델 정의
class MovieInfo(BaseModel):
    id: str
    title: str
    year: int
    popularity_log: float
    genres: List[str]
    str_genres_keywords: str


class RecommendedMovie(BaseModel):
    id: str
    title: str
    year: int
    popularity_log: float
    genres: List[str]
    str_genres_keywords: str


class RecommendationRequest(BaseModel):
    title: str
    top_k: Optional[int] = 30
    num_recommendations: Optional[int] = 10


class RecommendationResponse(BaseModel):
    input_movie: MovieInfo
    recommended_movies: List[RecommendedMovie]
    total_recommendations: int


class MovieListResponse(BaseModel):
    total_movies: int
    movies: List[str]


# Lifespan 이벤트로 앱 시작 시 데이터 로딩
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 데이터 로딩
    global movies, sorted_similarity_of_euclidean, tfidf_vectorizer
    
    logger.info("애플리케이션 시작 - 데이터 초기화 중...")
    movies_raw = load_data()
    movies = preprocess_data(movies_raw)
    sorted_similarity_of_euclidean, tfidf_vectorizer = calculate_tfidf_similarity(movies)
    logger.info("데이터 초기화 완료")
    
    yield
    
    # 종료 시 정리 작업 (필요한 경우)
    logger.info("애플리케이션 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="영화추천서비스 API",
    description="TF-IDF와 유클리드 거리를 사용한 영화 추천 시스템 API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 정보 반환
    """
    return {
        "message": "영화추천서비스 API에 오신 것을 환영합니다!",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API 정보",
            "GET /movies": "영화 목록 조회",
            "GET /movies/{title}": "특정 영화 정보 조회",
            "POST /recommend": "영화 추천 (POST)",
            "GET /recommend/{title}": "영화 추천 (GET)",
            "GET /docs": "API 문서 (Swagger UI)",
            "GET /redoc": "API 문서 (ReDoc)"
        }
    }


@app.get("/movies", response_model=MovieListResponse)
async def get_movies():
    """
    전체 영화 목록 조회
    """
    if movies is None:
        raise HTTPException(status_code=503, detail="데이터가 아직 로딩되지 않았습니다.")
    
    movie_titles = sorted(movies['title'].unique().tolist())
    return MovieListResponse(
        total_movies=len(movie_titles),
        movies=movie_titles
    )


@app.get("/movies/{title}", response_model=MovieInfo)
async def get_movie_info(title: str):
    """
    특정 영화의 상세 정보 조회
    """
    if movies is None:
        raise HTTPException(status_code=503, detail="데이터가 아직 로딩되지 않았습니다.")
    
    movie = movies[movies['title'] == title]
    
    if movie.empty:
        raise HTTPException(status_code=404, detail=f"'{title}' 제목의 영화를 찾을 수 없습니다.")
    
    movie_info = movie.iloc[0]
    
    return MovieInfo(
        id=str(movie_info['id']),
        title=movie_info['title'],
        year=int(movie_info['year']),
        popularity_log=float(movie_info['popularity_log']),
        genres=movie_info['genres'] if isinstance(movie_info['genres'], list) else [],
        str_genres_keywords=str(movie_info['str_genres_keywords']) if pd.notna(movie_info['str_genres_keywords']) else ""
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_movies(request: RecommendationRequest):
    """
    영화 추천 (POST 방식)
    """
    if movies is None or sorted_similarity_of_euclidean is None:
        raise HTTPException(status_code=503, detail="데이터가 아직 로딩되지 않았습니다.")
    
    recommended_movies, movie_info = recomm_of_euclidean(
        movies,
        sorted_similarity_of_euclidean,
        request.title,
        top_k=request.top_k,
        num_recommendations=request.num_recommendations
    )
    
    if recommended_movies is None or movie_info is None:
        raise HTTPException(status_code=404, detail=f"'{request.title}' 제목의 영화를 찾을 수 없습니다.")
    
    # 입력 영화 정보
    input_movie = MovieInfo(
        id=str(movie_info['id']),
        title=movie_info['title'],
        year=int(movie_info['year']),
        popularity_log=float(movie_info['popularity_log']),
        genres=movie_info['genres'] if isinstance(movie_info['genres'], list) else [],
        str_genres_keywords=str(movie_info['str_genres_keywords']) if pd.notna(movie_info['str_genres_keywords']) else ""
    )
    
    # 추천 영화 목록
    recommended_list = []
    for _, movie in recommended_movies.iterrows():
        recommended_list.append(RecommendedMovie(
            id=str(movie['id']),
            title=movie['title'],
            year=int(movie['year']),
            popularity_log=float(movie['popularity_log']),
            genres=movie['genres'] if isinstance(movie['genres'], list) else [],
            str_genres_keywords=str(movie['str_genres_keywords']) if pd.notna(movie['str_genres_keywords']) else ""
        ))
    
    return RecommendationResponse(
        input_movie=input_movie,
        recommended_movies=recommended_list,
        total_recommendations=len(recommended_list)
    )


@app.get("/recommend/{title}", response_model=RecommendationResponse)
async def recommend_movies_get(
    title: str,
    top_k: int = 30,
    num_recommendations: int = 10
):
    """
    영화 추천 (GET 방식)
    
    Query Parameters:
    - top_k: 후보 영화 개수 (기본값: 30)
    - num_recommendations: 최종 추천 영화 개수 (기본값: 10)
    """
    if movies is None or sorted_similarity_of_euclidean is None:
        raise HTTPException(status_code=503, detail="데이터가 아직 로딩되지 않았습니다.")
    
    recommended_movies, movie_info = recomm_of_euclidean(
        movies,
        sorted_similarity_of_euclidean,
        title,
        top_k=top_k,
        num_recommendations=num_recommendations
    )
    
    if recommended_movies is None or movie_info is None:
        raise HTTPException(status_code=404, detail=f"'{title}' 제목의 영화를 찾을 수 없습니다.")
    
    # 입력 영화 정보
    input_movie = MovieInfo(
        id=str(movie_info['id']),
        title=movie_info['title'],
        year=int(movie_info['year']),
        popularity_log=float(movie_info['popularity_log']),
        genres=movie_info['genres'] if isinstance(movie_info['genres'], list) else [],
        str_genres_keywords=str(movie_info['str_genres_keywords']) if pd.notna(movie_info['str_genres_keywords']) else ""
    )
    
    # 추천 영화 목록
    recommended_list = []
    for _, movie in recommended_movies.iterrows():
        recommended_list.append(RecommendedMovie(
            id=str(movie['id']),
            title=movie['title'],
            year=int(movie['year']),
            popularity_log=float(movie['popularity_log']),
            genres=movie['genres'] if isinstance(movie['genres'], list) else [],
            str_genres_keywords=str(movie['str_genres_keywords']) if pd.notna(movie['str_genres_keywords']) else ""
        ))
    
    return RecommendationResponse(
        input_movie=input_movie,
        recommended_movies=recommended_list,
        total_recommendations=len(recommended_list)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
