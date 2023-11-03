from sklearn import datasets, svm, model_selection
from base import TrainModelByMlflow

def train_iris_model():
  dataset  = datasets.load_iris()
  model = svm.SVC()

  # parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
  parameters = {"kernel": ("sigmoid", "poly")}

  clf = model_selection.GridSearchCV(model, parameters, cv=5)

  mlflow_object = TrainModelByMlflow(clf, dataset.data, dataset.target)
  mlflow_object.train()


if __name__ == "__main__":
  train_iris_model()