
import streamlit as st
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    """데이터 로딩 및 전처리"""
    movies_metadata = pd.read_csv('data/movies_metadata.csv')
    links_small = pd.read_csv('data/links_small.csv')

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


@st.cache_data
def compute_similarity(_movies):
    """유사도 계산"""
    bow_vectorizer = CountVectorizer()
    genre_mat = bow_vectorizer.fit_transform(_movies['str_genres'])
    similarity_of_genre = cosine_similarity(genre_mat, genre_mat)
    sorted_similarity_of_genre = similarity_of_genre.argsort()[:, ::-1]
    return similarity_of_genre, sorted_similarity_of_genre


def recommendation(movies, sorted_similarity_of_genre, similarity_of_genre, title_name, top_k=5):
    """영화 추천 함수"""
    movie_of_title = movies[movies['title'] == title_name]

    if len(movie_of_title) == 0:
        return None, None

    genre = movie_of_title['str_genres'].values[0]
    movie_index_of_title = movie_of_title.index.values[0]

    similar_indexes = sorted_similarity_of_genre[movie_index_of_title, :(top_k*2)]
    similar_indexes = similar_indexes.reshape(-1)
    similar_indexes = similar_indexes[similar_indexes != movie_index_of_title]

    result = movies.iloc[similar_indexes].copy()
    # 유사도 점수 추가
    result['similarity'] = [similarity_of_genre[movie_index_of_title, idx] for idx in similar_indexes]
    result = result.sort_values(by=['year'], ascending=False)[:top_k]

    return result, genre


# Streamlit UI
st.title("영화 추천 시스템")
st.subheader("장르 기반 유사 영화 추천")

# 데이터 로딩 (session_state로 한 번만 로딩)
if 'data_loaded' not in st.session_state:
    with st.spinner("데이터 로딩 중... (최초 1회만 실행)"):
        st.session_state.movies = load_data()
        st.session_state.similarity_of_genre, st.session_state.sorted_similarity_of_genre = compute_similarity(st.session_state.movies)
        st.session_state.data_loaded = True

movies = st.session_state.movies
similarity_of_genre = st.session_state.similarity_of_genre
sorted_similarity_of_genre = st.session_state.sorted_similarity_of_genre

st.success(f"총 {len(movies)}개의 영화 데이터 로딩 완료!")

# 영화 제목 입력
movie_titles = movies['title'].tolist()
selected_movie = st.selectbox("영화를 선택하세요:", movie_titles)

# Top K 설정
top_k = st.slider("추천 영화 개수", min_value=1, max_value=10, value=5)

# 추천 버튼
if st.button("추천 영화 찾기"):
    with st.spinner("유사 영화 검색 중..."):
        result, genre = recommendation(movies, sorted_similarity_of_genre, similarity_of_genre, selected_movie, top_k)

    if result is not None:
        st.markdown(f"### 선택한 영화: **{selected_movie}**")
        st.markdown(f"**장르:** {genre}")

        st.markdown("---")
        st.markdown(f"### 유사한 영화 Top {top_k}")

        # 결과 테이블 표시
        display_df = result[['title', 'str_genres', 'year', 'popularity', 'similarity']].copy()
        display_df.columns = ['제목', '장르', '개봉년도', '인기도', '유사도']
        display_df['유사도'] = display_df['유사도'].apply(lambda x: f"{x:.2%}")
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1

        st.dataframe(display_df, use_container_width=True)

        # 추천 영화 카드 형식으로 표시
        st.markdown("---")
        st.markdown("### 추천 영화 상세")

        for idx, row in result.iterrows():
            with st.expander(f"{row['title']} ({int(row['year'])})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**장르:** {row['str_genres']}")
                with col2:
                    st.write(f"**인기도:** {row['popularity']:.1f}")
    else:
        st.error("영화를 찾을 수 없습니다.")
