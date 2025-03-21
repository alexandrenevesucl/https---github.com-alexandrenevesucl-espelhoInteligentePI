import os
import zipfile
import requests

# URL do dataset FER2013 no Kaggle
url = "https://www.kaggle.com/datasets/msambare/fer2013/download"

# Baixa o dataset (caso precise login, baixe manualmente e extraia)
file_path = "fer2013.zip"
if not os.path.exists(file_path):
    response = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(response.content)

# Extrai os arquivos
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall("dataset")
