import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset from CSV
path = 'C:/Users/Админ/Downloads/cleaned_movies_dataset.csv'
df = pd.read_csv(path)

# Streamlit app setup
st.title("Movie Recommendation System")
st.sidebar.header("Movie Selection")

search_query = st.sidebar.text_input("Search for a Movie", "")

filtered_movies = df[df['Title'].str.contains(search_query, case=False, na=False)] if search_query else df


# Movie selection dropdown
selected_movie = st.sidebar.selectbox("Select a Movie", filtered_movies['Title'].tolist())

if selected_movie:
    # Fetch selected movie details
    selected_movie_data = df[df['Title'] == selected_movie].iloc[0]
    genre = selected_movie_data['Genre(s)']
    release = selected_movie_data['Release Year']
    rating = selected_movie_data['IMDb Rating']
    description = selected_movie_data['Description']
    poster_url = selected_movie_data['Poster URL']
    

    # Display movie details
    st.write("### Selected Movie Details")
    st.write(f"**Title:** {selected_movie}")
    st.write(f"**Genre:** {genre}")
    st.write(f"**Release year:** {release}")
    st.write(f"**Rating:** {rating}")
    st.write(f"**Description:** {description}")
    st.image(poster_url)  # Display poster image using URL

    # Recommend movies using the content-based method
    df['combined_features'] = df['Genre(s)'] + " " + df['Description']

    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(feature_matrix)




    def recommend_movies(title, similarity_matrix, top_n=13):
        if not hasattr(similarity_matrix, "shape") or len(similarity_matrix.shape) != 2:
            raise ValueError("The similarity matrix is not a valid 2D array.")

    # Get the index of the selected movie
        movie_idx = df[df['Title'] == title].index[0]

    # Get similarity scores for the selected movie
        similarity_scores = list(enumerate(similarity_matrix[movie_idx]))

    # Debugging: Check similarity scores
        print(f"Similarity scores for '{title}': {similarity_scores}")

    # Sort movies by similarity scores
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Debugging: Check sorted scores
        print(f"Sorted scores: {sorted_scores}")

    # Get top recommendations (excluding the selected movie itself)
        recommended_indices = [i[0] for i in sorted_scores[1:top_n+1]]

    # Debugging: Check recommended indices
        print(f"Recommended indices: {recommended_indices}")

    # Return recommended movie details (using .iloc to fetch rows by position)
        return df.iloc[recommended_indices]

# Generate recommendations
recommendations = recommend_movies(selected_movie, similarity_matrix)

# Display recommended movies
st.write("### Recommended Movies")
for _, row in recommendations.iterrows():
    # Display movie poster and details
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(row['Poster URL'], width=100)  # Display poster image
    with col2:
        st.write(f"**Title:** {row['Title']}")
        st.write(f"**Genre:** {row['Genre(s)']}")
        st.write(f"**Release Year:** {row['Release Year']}")
        st.write(f"**Rating:** {row['IMDb Rating']}")

        # Add a "More" button to display description
        if st.button(f"More about {row['Title']}"):
            st.write(f"**Description:** {row['Description']}")
