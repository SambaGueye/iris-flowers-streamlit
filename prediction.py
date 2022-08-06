import joblib
def make_prediction(data):
    rfc = joblib.load("xgb_model.sav")
    return rfc.predict(data)