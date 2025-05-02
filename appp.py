import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit config
st.set_page_config(page_title="Course Name-Based Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation_v2.xlsx")

    # Encode categorical features
    le_diff = LabelEncoder()
    le_mat = LabelEncoder()
    df["difficulty_enc"] = le_diff.fit_transform(df["difficulty_level"])
    df["material_enc"] = le_mat.fit_transform(df["study_material_available"])

    # Normalize numeric columns
    scaler = MinMaxScaler()
    num_features = ['course_duration_hours', 'course_price', 'rating', 'feedback_score',
                    'time_spent_hours', 'previous_courses_taken', 'enrollment_numbers',
                    'difficulty_enc', 'material_enc']
    df[num_features] = scaler.fit_transform(df[num_features])

    return df, num_features

df, feature_cols = load_data()

# Create a mapping of unique course names to their mean feature vectors
@st.cache_data
def get_course_feature_matrix(df, features):
    grouped = df.groupby("course_name")[features].mean().reset_index()
    return grouped

course_features = get_course_feature_matrix(df, feature_cols)
available_courses = course_features["course_name"].unique()[:20]  # Limit to 20 for dropdown

# Streamlit UI
st.title("Course Recommender Based on Course Name")
selected_course = st.selectbox("Select a course name", available_courses)

def recommend_courses(selected_course, course_features, top_n=5):
    if selected_course not in course_features["course_name"].values:
        return pd.DataFrame()

    # Extract feature vector of selected course
    selected_vector = course_features[course_features["course_name"] == selected_course][feature_cols].values
    other_vectors = course_features[feature_cols].values

    # Compute cosine similarity
    similarities = cosine_similarity(selected_vector, other_vectors).flatten()
    course_features["similarity"] = similarities

    # Return top N similar courses excluding the selected one
    recs = course_features[course_features["course_name"] != selected_course]
    return recs.sort_values("similarity", ascending=False).head(top_n)[["course_name", "similarity"]]

if st.button("Recommend Courses"):
    with st.spinner("Generating recommendations..."):
        recs = recommend_courses(selected_course, course_features)
        if not recs.empty:
            st.subheader(f"Top recommendations similar to **{selected_course}**")
            st.dataframe(recs.reset_index(drop=True))
        else:
            st.warning("No recommendations found.")

# Optional: Show full raw data
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)
