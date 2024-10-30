from preprocess import preprocess_data
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def train_doc2vec(X_train):
    # Train the Doc2Vec model
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(X_train)]
    model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, workers=4)
    return model

def predict_doc2vec(model, X_test):
    # Inference with Doc2Vec model
    predictions = [model.infer_vector(doc.words) for doc in X_test]
    return predictions

def run_doc2vec(df, text_column, target_column):
    X_train, X_test, y_train, y_test = preprocess_data(df, text_column, target_column)
    model = train_doc2vec(X_train)
    predictions = predict_doc2vec(model, X_test)
    return predictions, y_test
