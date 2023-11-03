from sklearn import datasets, svm, model_selection
from src.base import TrainModelByMlflow

def train_iris_model():
  '''
  모델을 실제로 훈련시키는 research workspace 입니다.
  해당 함수는 Iris datset 을 이용해 훈련하는 상황을 가정했습니다.
  '''
  dataset  = datasets.load_iris()
  model = svm.SVC()

  # parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
  parameters = {"kernel": ("sigmoid", "poly")}

  clf = model_selection.GridSearchCV(model, parameters, cv=5)

  mlflow_object = TrainModelByMlflow(clf, dataset.data, dataset.target)
  mlflow_object.train()


if __name__ == "__main__":
  train_iris_model()