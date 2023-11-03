import sqlite3
import os
import numpy as np
from PIL import Image
import io
from sklearn import datasets

def create_database():
  # 데이터베이스 연결 생성 (파일이 없으면 새로 생성됨)
  conn = sqlite3.connect('mlflow.db')
  # 커서 객체 생성
  cursor = conn.cursor()
  # 테이블 생성
  cursor.execute('''CREATE TABLE IF NOT EXISTS images
                    (id integer PRIMARY KEY AUTOINCREMENT, image blob, label integer)''')
  # 변경사항 커밋
  conn.commit()
  # 연결 종료
  conn.close()

def load_image_as_blob():

  conn = sqlite3.connect('mlflow.db')
  cursor = conn.cursor()

  labels = [0,1]
  
  for label in labels:
    dir_list = os.listdir(f"data/{label}")
    for image in dir_list:
      with open(f'data/{label}/{image}', 'rb') as file:
        image_data = file.read()
      cursor.execute('''INSERT INTO images (image, label) VALUES (?, ?)''', (image_data, label))
    
  conn.commit()
  conn.close()


def get_images_by_label(label:int):
  conn = sqlite3.connect('mlflow.db')
  cursor = conn.cursor()
  cursor.execute(f'''select image.image 
                 from images as image
                 where image.label = {label}
                 limit 10''')
  image_arrays = []
  for row in cursor.fetchall():
      image_data = row[0]
      # 바이너리 스트림으로 이미지 데이터 읽기
      image = Image.open(io.BytesIO(image_data))
      # 이미지를 NumPy 배열로 변환
      image_array = np.hstack(np.array(image))
      image_arrays.append(image_array)
  conn.close()
  return image_arrays



