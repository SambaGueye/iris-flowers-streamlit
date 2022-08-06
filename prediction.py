import joblib
def make_prediction(data):
    clf = joblib.load("xgb_model.pkl")
    return clf.predict(data)