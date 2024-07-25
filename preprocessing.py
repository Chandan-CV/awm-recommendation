from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np



def get_features_array(data):

    text_columns = ["I define privilege to be (having a supportive family, dependable friends, a job you love, anything else)",
    "About my growing up years  (they were awesome, they were challenging but all is good now, about the family, about friends, memories etc etc)",
    "Anything else that you may like to mention to the community members (any specifics which may be important to you)"]
    categorical_columns = ["My city of residence", "My current relationship status", "I have children",
                        "How do I introduce the professional me (banker, IT professional, entrepreneur, environmentalist)",
                        "Name of the institute I graduated from last.", "Educational degree",
                        'Ball park of my professional annual income']
    numerical_columns = ['My age']


    age_weight = 3.0
    relationship_status = 2.0
    city_weight = 1.5


    # Vectorize text columns using TF-IDF
    tfidf_vectorizers = {}
    tfidf_features = []

    for column in text_columns:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data[column])
        tfidf_vectorizers[column] = vectorizer
        tfidf_features.append(tfidf_matrix)

    label_encoders = {}
    encoded_categorical_features = []

    for column in categorical_columns:
        le = LabelEncoder()
        encoded_column = le.fit_transform(data[column])
        label_encoders[column] = le
        if(column == 'My city of residence'):
            encoded_categorical_features.append(encoded_column.reshape(-1, 1)*city_weight)
        elif(column == 'My current relationship status'):
            encoded_categorical_features.append(encoded_column.reshape(-1, 1)*relationship_status)
        else:
            encoded_categorical_features.append(encoded_column.reshape(-1, 1))


    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(data[numerical_columns])
    scaled_numerical_features[:, 0] *= age_weight

    # Combine all features into a single feature matrix
    features = hstack(tfidf_features + encoded_categorical_features + [scaled_numerical_features])

    return features



def preprocessData(dataset):
    dataset = (dataset.iloc[:,:3]).join(dataset.iloc[:,6])

    vectorizer = TfidfVectorizer()
    city_vectorized = vectorizer.fit_transform(dataset['My city of residence'])

    # one hot encoding
    encoded_categorical_features =  (pd.get_dummies(dataset['My current relationship status'], dtype=int))

    #label encode
    labelencoder = LabelEncoder()
    labeledFeatures = labelencoder.fit_transform(dataset['My Gender'])

    scaler = StandardScaler()
    scaled_numerical_features= scaler.fit_transform((dataset.iloc[:,1].values).reshape(-1,1))

    labeledFeatures=labeledFeatures.reshape(-1,1)
    city_vectorized = city_vectorized.toarray()

    features = np.hstack([labeledFeatures, scaled_numerical_features, encoded_categorical_features.values, city_vectorized])
    return features
