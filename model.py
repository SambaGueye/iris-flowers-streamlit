from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as rfc
import joblib as joblib

iris = load_iris()

if __name__=='__main__':
    numSamples, numFeatures = iris.data.shape

    # print(iris.target_names)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
    
    model = rfc(max_depth=5).fit(X_train, y_train)

    predictions = model.predict(X_test)

    # print(model.score(X_test, y_test))
    joblib.dump(model, "xgb_model.sav")
