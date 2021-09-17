from django.shortcuts import render
import cv2
import os
import pickle
import base64
import json

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from django.http import JsonResponse
from django.http import HttpResponse
# Create your views here.

CATEGORIES = ['orange', 'Violet', 'red', 'Blue', 'Green', 'Black', 'Brown', 'White']
IMG_SIZE = 100

def train(request):
    """
    There is a problem with this implementation. I could not find a way to upload my own image for prediction.
    """
    #https://www.kaggle.com/ashutoshvarma/image-classification-using-svm-92-accuracy/notebook
    print("inside soil")
    DATADIR = str(Path(__file__).resolve().parent.parent) + '/soil/ColorClassification/ColorClassification'

    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    lenofimage = len(training_data)

    X = []
    y = []
    for categories, label in training_data:
        X.append(categories)
        y.append(label)
    X = np.array(X).reshape(lenofimage, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = SVC(kernel='linear', gamma='auto')
    model.fit(X_train, y_train)

    y2 = model.predict(X_test)

    # TESTDATADIR = str(Path(__file__).resolve().parent.parent) + '/soil/TestColor'
    # data = ""
    # for img in os.listdir(TESTDATADIR):
    #     try:
    #         img_array = cv2.imread(os.path.join(TESTDATADIR, img))
    #         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #         l = np.array(new_array).reshape(lenofimage, -1)
    #         data = model.predict(l)
    #         print(data)
    #     except Exception as e:
    #         print(e)

    return HttpResponse(f'Total training data length: {len(training_data)}. X.Shape: {X.shape}'
                        f'Accuracy on unknown data is,{accuracy_score(y_test,y2)}')

def train_using_github_code(request):
    #https://github.com/ShanmukhVegi/Image-Classification/blob/main/Shanmukh_Classification.ipynb

    DATADIR = str(Path(__file__).resolve().parent.parent) + '/soil/ColorClassification/ColorClassification'
    flat_data_arr = []
    target_arr = []
    for category in CATEGORIES:
        print(f'loading... category : {category}')
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(CATEGORIES.index(category))
        print(f'loaded category:{category} successfully')
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    #splitting
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77)
    print('Splitted Successfully')

    #training
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
    svc = svm.SVC(probability=True)
    print("The training of the model is started, please wait for while as it may take few minutes to complete")
    model = GridSearchCV(svc, param_grid)
    model.fit(x_train, y_train)
    print('The Model is trained well with the given images')

    y_pred = model.predict(x_test)

    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")

    pickle.dump(model, open('img_model.p', 'wb'))
    print("Pickle is dumped successfully")

    print("done")
    return accuracy_score(y_pred, y_test) * 100
    # return HttpResponse(f'The model is {accuracy_score(y_pred, y_test) * 100}% accurate')

def find_probability(request):
    TESTDATADIR = str(Path(__file__).resolve().parent.parent) + '/soil/TestColor'
    model = pickle.load(open('img_model.p', 'rb'))
    for img in os.listdir(TESTDATADIR):
        try:
            img_array = imread(os.path.join(TESTDATADIR, img))
            img_resize = resize(img_array, (150, 150, 3))
            l = [img_resize.flatten()]
            probability = model.predict_proba(l)
            # print(probability)
            for ind, val in enumerate(CATEGORIES):
                print(f'{val} = {probability[0][ind] * 100}%')
            print("The predicted image is : " + CATEGORIES[model.predict(l)[0]])
        except Exception as e:
            print(e)


def index(request):
    return render(request, "base.html", {})


def train_page(request):
    return render(request, "train_page.html", {})


def test_page(request):
    return render(request, "test_page.html", {})


def testing(request):
    image = request.body.decode("UTF-8")
    json_image = json.loads(image)
    TESTDATADIR = str(Path(__file__).resolve().parent.parent) + '/soil/TestColor'
    TESTDATADIR = TESTDATADIR + '/image.jpg'
    decodeit = open(TESTDATADIR, 'wb')
    decodeit.write(base64.b64decode((json_image['image'])))
    decodeit.close()

    TESTDATADIR = str(Path(__file__).resolve().parent.parent) + '/soil/TestColor'
    model = pickle.load(open('img_model.p', 'rb'))
    data_to_send = ""
    for img in os.listdir(TESTDATADIR):
        try:
            img_array = imread(os.path.join(TESTDATADIR, img))
            img_resize = resize(img_array, (150, 150, 3))
            l = [img_resize.flatten()]
            probability = model.predict_proba(l)
            for ind, val in enumerate(CATEGORIES):
                print(f'{val} = {probability[0][ind] * 100}%')
            print("The predicted color is : " + CATEGORIES[model.predict(l)[0]])
            data_to_send = "The predicted color is : " + CATEGORIES[model.predict(l)[0]]
        except Exception as e:
            print(e)
    print(data_to_send)
    response = JsonResponse({'message': data_to_send})

    return response


def start_training(request):
    print("hererere")
    accuracy = train_using_github_code()
    # time.sleep(3)
    response = JsonResponse({'message': 'done'})
    return response
