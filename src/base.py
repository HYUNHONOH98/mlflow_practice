import sys
sys.path.append(".")

from pprint import pprint
from src.utils import fetch_logged_data
from mlflow.entities import ViewType
from mlflow import MlflowClient
import mlflow
import numpy as np


class MlflowBase:
    '''
    mlflow 트래킹을 활성화 하기 위한 베이스 클래스 입니다.
    '''
    def __init__(self):
        self.tracking_uri = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri(self.tracking_uri)


class TrainModelByMlflow(MlflowBase):
    '''
    mlflow 를 이용한 training 을 범용적인 모델에 대해 시행할 수 있는 클래스입니다.
    (대부분의 scikit-learn 등의 모델을 이용할 수 있는 형태)
    '''
    def __init__(self, model, X, y=None):
        super().__init__()
        self.model = model
        self.X = X
        self.y = y

    def train(self):
        mlflow.autolog()

        self.model.fit(self.X, self.y)
        run_id = mlflow.last_active_run().info.run_id
        print(f"Logged data and model in run {run_id}")

        # show logged data
        for key, data in fetch_logged_data(run_id).items():
            print(f"\n---------- logged {key} ----------")
            pprint(data)


class SearchModelByMlflow(MlflowBase):
    '''
    생성한 모델 중 best 모델을 찾는 클래스 입니다.
    '''
    def __init__(self):
        super().__init__()

    def get_best_model(self):
        
        run = MlflowClient().search_runs(
            experiment_ids=["0"],
            filter_string="",
            run_view_type=ViewType.ALL,
            max_results=1,
            order_by=["metrics.best_cv_score DESC"],
        )[0]
        return {"artifact_uri":run.info.artifact_uri,
                "score":run.data.metrics}
    

def model_path(artifact_uri : str):
    return artifact_uri.replace("mlflow-artifacts:", "mlartifacts") + "/model"


class InferenceModelByMlflow(SearchModelByMlflow):
    '''
    생성한 모델 중 best 모델을 inference 할 수 있게 만든 클래스 입니다.
    '''
    def __init__(self):
        super().__init__()
        self.best_artifact_uri = self.get_best_model()["artifact_uri"]

    def inference_best_model(self, data):
        model = mlflow.sklearn.load_model(model_path(self.best_artifact_uri))
        return model.predict(np.array([data]))