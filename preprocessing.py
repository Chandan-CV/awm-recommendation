from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, StandardScaler




def get_features_array(data):
    text_columns = ["I define privilege to be (having a supportive family, dependable friends, a job you love, anything else)",
    "About my growing up years  (they were awesome, they were challenging but all is good now, about the family, about friends, memories etc etc)",
    "Anything else that you may like to mention to the community members (any specifics which may be important to you)"]
    categorical_columns = ["My city of residence", "My current relationship status", "I have children",
                        "How do I introduce the professional me (banker, IT professional, entrepreneur, environmentalist)",
                        "Name of the institute I graduated from last.", "Educational degree",
                        'Ball park of my professional annual income']
    numerical_columns = ['My age']

    # Vectorize text columns using TF-IDF
    tfidf_vectorizers = {}
    tfidf_features = []

    for column in text_columns:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data[column])
        tfidf_vectorizers[column] = vectorizer
        tfidf_features.append(tfidf_matrix)

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    encoded_categorical_features = []

    for column in categorical_columns:
        le = LabelEncoder()
        encoded_column = le.fit_transform(data[column])
        label_encoders[column] = le
        encoded_categorical_features.append(encoded_column.reshape(-1, 1))

    # Standardize numerical features
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(data[numerical_columns])

    # Combine all features into a single feature matrix
    features = hstack(tfidf_features + encoded_categorical_features + [scaled_numerical_features])

    return features