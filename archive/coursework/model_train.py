import preprocessing
from sklearn.pipeline import Pipeline

def train(model,X_train,y_train,params={}):
    pipeline = preprocessing.preprocess_pipeline()
    train_model = Pipeline([('preprocess',pipeline),('model',model(**params))])
    train_model.fit(X_train,y_train)
    return train_model