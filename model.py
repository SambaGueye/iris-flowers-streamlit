from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier as xgb
from sklearn.metrics import accuracy_score
import joblib as joblib

iris = load_iris()

if __name__=='__main__':
    numSamples, numFeatures = iris.data.shape

    # print(iris.target_names)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
    
    model = xgb(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(model.score(X_test, y_test))
    joblib.dump(model, "xgb_model.pkl")
