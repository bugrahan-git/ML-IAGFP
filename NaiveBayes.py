from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from math import copysign, log10, sqrt
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
from random import randrange
from csv import reader
from GIST import GIST
from PIL import Image
import pandas as pd
import numpy as np
import time
import os
import cv2
import time



def get_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def get_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


def get_gist(input_image):
    image = np.array(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY))
    gist = GIST(param)
    gist_feature = gist._gist_extract(image)
    return np.array(gist_feature)


def prepare_data():
    movements = os.listdir("dataset")
    for movement in movements:
        artists = os.listdir("dataset/"+movement)
        for artist in artists:
            images = os.listdir("dataset/"+movement+"/"+artist)
            for image in images:
                img = cv2.imread("dataset/"+movement+"/"+artist+"/"+image)
                if img is not None:
                    img = cv2.resize(img, fixed_size)
                    image_list.append(img)
                    artist_list.append(artist)
                    movement_list.append(movement)
                else:
                    print(image, " None")
               


def prepare_mini_data():
    artists = os.listdir("dataset/impressionism")
    for artist in artists:
        images = os.listdir("dataset/impressionism/"+artist)
        for image in images:
            img = cv2.imread("dataset/impressionism/"+artist+"/"+image)
            if img is not None:
                img = cv2.resize(img, fixed_size)
                image_list.append(img)
                artist_list.append(artist)
            else:
                print(image, " None")


    
def get_feature_vec():
    p = Pool()
    print("in pool")
    hu_moments_list = p.map(get_hu_moments, image_list)
    histogram_list = p.map(get_histogram, image_list)
    gist_list = p.map(get_gist, image_list)
    feature_vec = np.hstack([np.vstack(hu_moments_list), np.vstack(histogram_list), np.vstack(gist_list)])
    return feature_vec

def get_his_vec():
    p = Pool()
    histogram_list = p.map(get_histogram, image_list)
    feature_vec = np.vstack(histogram_list)
    return feature_vec


"""
INITIALIZATION PARAMETERS FOR GIST AND COLOR HISTOGRAM
"""
bins = 8
param = {
        "orientationsPerScale":np.array([8,8]),
        "numberBlocks":[10,10],
        "fc_prefilt":10,
        "boundaryExtension":32
}
fixed_size = tuple((250, 250))

image_list = list()
artist_list = list()
movement_list = list()
feature_list = list()


prepare_data()


X = get_feature_vec()
y = np.array(artist_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


print("ARTIST CLASSIFICATION")
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
score = accuracy_score(y_test, y_pred)
print("Score = ", score)


print("--------------------------")
print("MOVEMENT CLASSIFICATION")
X = np.array(feature_list)
y = np.array(movement_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
score = accuracy_score(y_test, y_pred)
print("Score = ", score)