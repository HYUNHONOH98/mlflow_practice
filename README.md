# mlflow_practice

### MLflow practice for likelion seminar

- 가상 환경 생성

  - python -m venv .venv

- 가상 환경 활성화 for mac
  - source .venv/bin/activate
- 가상 환경 활성화 for window

  - source .venv\Scripts\activate
 
- 패키지 다운로드
  - pip install -r requirements.txt 

- mlflow 서버 켜기 (port 5000)

  - mlflow server --backend-store-uri sqlite:///mlflow.db

- fastapi 서버 켜기 (port 8000)
  - uvicorn main:app --host 0.0.0.0 --port 8000
  - or
  - uvicorn main:app --reload

```
mlflow_practice
├─ .gitignore
├─ db
│  ├─ __init__.py
│  └─ crud.py
├─ src
│  ├─ __init__.py
│  ├─ base.py
│  └─ utils.py
├─ workspace
│  └─ train_iris.py
├─ main.py
└─  requirements.txt
```
