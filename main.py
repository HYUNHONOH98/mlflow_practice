from fastapi import FastAPI
from src.base import SearchModelByMlflow, InferenceModelByMlflow
from pydantic import BaseModel

app = FastAPI()

@app.get("/best_model")
def get_best_model():
  '''
  metric 종류 : mean_test_score , std_test_score
  '''
  run_info = SearchModelByMlflow().get_best_model()
  return run_info

class IrisData(BaseModel):
  data: list

@app.post("/best_model/{task_id}")
def inference_iris_best_model(iris_data: IrisData):
  result = InferenceModelByMlflow().inference_best_model(iris_data.data)
  return {"result" : int(result[0])}
  