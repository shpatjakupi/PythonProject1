from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import diabetes_corr as dc
from pandas import DataFrame as df
import matplotlib.pyplot as plt


def train_data(newdf):
    feature_col_names = ['Pregnancies', 'Glucose', 'Bloodpressure', 'Insulin', 'Bodymass', 'Diabetes_pedigree_function', 'Age']
    predicted_class_names = ['Class']

    X = newdf[feature_col_names].values
    y = newdf[predicted_class_names].values

    split_test_size = 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def naive_bayes(X_train, y_train):
    nb_model = GaussianNB()

    nb_model.fit(X_train, y_train.ravel())
    return nb_model


def models_acuracy(X_train, nb_model):
    nb_predict_train = nb_model.predict(X_train)
    return nb_predict_train


def models_acuracy_X_Test(X_test, nb_model):
    nb_predict_test = nb_model.predict(X_test)
    return nb_predict_test
