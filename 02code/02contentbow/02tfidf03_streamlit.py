import streamlit as st
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances


@st.cache_data
def load_data(metadata_path="data/movies_metadata.csv", 
              links_path="data/links_small.csv", 
              keywords_path="data/keywords.csv"):
    """
    영화 데이터를 로드하고 병합하는 함수 (Streamlit 캐싱 적용)
    
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


@st.cache_data
def preprocess_data(movies):
    """
    영화 데이터를 전처리하는 함수 (Streamlit 캐싱 적용)
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


@st.cache_data
def calculate_tfidf_similarity(movies):
    """
    TF-IDF 벡터화를 수행하고 유클리드 거리 기반 유사도를 계산하는 함수 (Streamlit 캐싱 적용)
    
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


def main():
    """
    Streamlit 메인 애플리케이션
    """
    # 페이지 설정
    st.set_page_config(
        page_title="영화추천서비스 시스템",
        page_icon="🎬",
        layout="wide"
    )
    
    # Flex 레이아웃을 위한 CSS 스타일
    st.markdown("""
    <style>
    /* 검색 영역 flex 스타일 개선 */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 제목 및 설명
    st.title("영화추천서비스 시스템")
    st.markdown("---")
    st.markdown("""
    이 애플리케이션은 **TF-IDF**와 **유클리드 거리**를 사용하여 영화를 추천합니다.
    좋아하는 영화 제목을 입력하면, 장르와 키워드가 유사한 영화들을 추천해드립니다.
    """)
    
    # 사이드바: 데이터 로딩 상태 표시
    with st.sidebar:
        st.header("📊 데이터 로딩")
        with st.spinner("데이터를 로딩하는 중..."):
            movies_raw = load_data()
            movies = preprocess_data(movies_raw)
            sorted_similarity_of_euclidean, tfidf_vectorizer = calculate_tfidf_similarity(movies)
        
        st.success(f"✅ {len(movies)}개의 영화 데이터 로딩 완료!")
        st.markdown("---")
        
        st.header("⚙️ 설정")
        num_recommendations = st.slider(
            "추천 영화 개수",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
    
    # 메인 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 영화 검색 및 추천")
        
        # 영화 제목 입력과 버튼을 Flex로 묶기
        movie_titles = sorted(movies['title'].unique().tolist())
        
        # 컬럼을 사용하여 flex 레이아웃 구현 (Streamlit의 컬럼은 자동으로 flex 적용)
        search_col1, search_col2 = st.columns([4, 1])
        
        with search_col1:
            selected_movie = st.selectbox(
                "추천받고 싶은 영화를 선택하세요:",
                options=movie_titles,
                index=0 if 'Jumanji' in movie_titles else 0,
                help="드롭다운에서 영화를 선택하거나 직접 입력할 수 있습니다.",
                label_visibility="visible"
            )
        
        with search_col2:
            st.write("")  # 버튼을 selectbox와 같은 높이로 맞추기 위한 공백
            search_button = st.button("🎯 추천 받기", type="primary", use_container_width=True)
        
        # 검색 버튼 클릭 처리
        if search_button:
            if selected_movie:
                with st.spinner("추천 영화를 계산하는 중..."):
                    recommended_movies, movie_info = recomm_of_euclidean(
                        movies,
                        sorted_similarity_of_euclidean,
                        selected_movie,
                        top_k=30,
                        num_recommendations=num_recommendations
                    )
                    
                    if recommended_movies is not None and movie_info is not None:
                        # 입력 영화 정보 표시
                        st.markdown("---")
                        st.subheader(f"📽️ 선택한 영화: {selected_movie}")
                        
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("연도", int(movie_info['year']))
                        with col_info2:
                            st.metric("인기도 (로그)", f"{movie_info['popularity_log']:.2f}")
                        with col_info3:
                            genres_str = ", ".join(movie_info['genres']) if isinstance(movie_info['genres'], list) else "N/A"
                            st.write(f"**장르:** {genres_str}")
                        
                        st.write(f"**장르/키워드:** {movie_info['str_genres_keywords']}")
                        
                        # 추천 영화 표시
                        st.markdown("---")
                        st.subheader(f"🎬 추천 영화 {num_recommendations}개")
                        
                        # 추천 영화를 카드 형태로 표시
                        for idx, (_, movie) in enumerate(recommended_movies.iterrows(), 1):
                            with st.expander(f"#{idx}. {movie['title']} ({int(movie['year'])})", expanded=False):
                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.write(f"**연도:** {int(movie['year'])}")
                                with col_m2:
                                    st.write(f"**인기도 (로그):** {movie['popularity_log']:.2f}")
                                with col_m3:
                                    genres_str = ", ".join(movie['genres']) if isinstance(movie['genres'], list) else "N/A"
                                    st.write(f"**장르:** {genres_str}")
                                st.write(f"**장르/키워드:** {movie['str_genres_keywords']}")
                        
                        # 추천 영화를 테이블로 표시
                        st.markdown("---")
                        st.subheader("📋 추천 영화 요약")
                        display_df = recommended_movies[['title', 'year', 'popularity_log']].copy()
                        display_df['year'] = display_df['year'].astype(int)
                        display_df['popularity_log'] = display_df['popularity_log'].round(2)
                        display_df.columns = ['제목', '연도', '인기도(로그)']
                        display_df.index = range(1, len(display_df) + 1)
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.error(f"'{selected_movie}' 제목의 영화를 찾을 수 없습니다.")
            else:
                st.warning("영화 제목을 입력해주세요.")
    
    with col2:
        st.subheader("📈 통계 정보")
        st.metric("전체 영화 수", len(movies))
        st.metric("평균 인기도 (로그)", f"{movies['popularity_log'].mean():.2f}")
        st.metric("평균 연도", f"{movies['year'].mean():.0f}")
        
        st.markdown("---")
        st.subheader("📊 인기도 분포")
        st.bar_chart(movies['popularity_log'].value_counts().head(10))
        
        st.markdown("---")
        st.subheader("📅 연도별 분포")
        year_counts = movies['year'].value_counts().sort_index().tail(20)
        st.bar_chart(year_counts)


if __name__ == "__main__":
    main()
