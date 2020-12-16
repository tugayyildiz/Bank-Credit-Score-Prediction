import pandas as pd
import zipfile
import urllib
import os

def get_data(folder_name="dataset", file_name="dataset.zip"):
    url = "https://github.com/recep-yildirim/Bank-Credit-Score-Prediction/blob/master/archive.zip?raw=true"
    path = os.path.join(folder_name, file_name)

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print("'{}' folder was created.".format(folder_name))

    else:
        print("Dataset will downloaded to exist '{}' folder.".format(folder_name))

    while os.path.isfile(path):
        file_name = str(input("File exist, enter a new file name: ")) + ".zip"
        path = os.path.join(folder_name, file_name)

    urllib.request.urlretrieve(url, path)

    with zipfile.ZipFile(path, "r") as dataset:
        dataset.extractall(folder_name)

    return pd.read_csv(folder_name + "/credit_train.csv"), pd.read_csv(folder_name + "/credit_test.csv")
