
# coding: utf-8

# # Imports

import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import re
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import os
import pandas as pd
from IPython.display import display, HTML,clear_output
from sklearn.neighbors import KNeighborsClassifier
import xml.etree.ElementTree as ET 
import imutils
import time
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.linear_model import LogisticRegression


# # Read and parse files


def open_bounding_boxes(train_dir):
    """
    Reads the annotation .xml files located in train_dir.
    There is one file per training sample that and includes: path of the image im_path, bounding box coordinates (xmin,ymin)(xmax,ymax),
    and bouding box label: label
    
    Returns: images_inside_box : List of all samples cropped at bouding box coordinates
             labels : List of labels for all samples
             images : Images (non cropped) for all samples
    
    """
    images_inside_box = [] # List of all samples cropped at bouding box coordinates
    labels = [] # List of labels for all samples
    images = [] # Images (non cropped) for all samples
    for filename in sorted(os.listdir(train_dir)): # Reads files in train_dir
        if not filename.endswith('.xml'): continue #For ex if it's jpeg 
        fullname = os.path.join(train_dir, filename)
        tree = ET.parse(fullname)
        im_path=tree.find('.//path').text #Tag path of .xml has image path
        
        for i,label in enumerate(tree.findall('.//name')): #Each sample may have more than one bounding box whose label        
            xmin = int(tree.findall('.//xmin')[i].text)    #is described in .xml's name tag
            ymin = int(tree.findall('.//ymin')[i].text)
            xmax = int(tree.findall('.//xmax')[i].text)
            ymax = int(tree.findall('.//ymax')[i].text)
            label=label.text
            try:
                img = cv2.imread(im_path)
            except:
                continue

            im_bound = img[ymin:ymax,xmin:xmax]
            im_bound = cv2.resize(im_bound, (0,0), fx=0.5, fy=0.5) #Resize the sample to half of its size 
            images_inside_box.append(im_bound)
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            labels.append(label)
        images.append(img)
        
    return images,images_inside_box,labels

def get_test_imgs(test_imgs):
    """
    Gets test samples which are not annotated 
    Returns: test_images : list of test samples
             list of test samples' path
    """
    test_images = []
    for img in sorted(os.listdir(test_imgs)):
        img = cv2.imread(test_imgs+img)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        test_images.append(img)
    return test_images,sorted(os.listdir(test_imgs))


# # Cards Model



class cards_model:
    """
    Class containing methods to preprocess, extract features, train and predict an image label
    Input : bow_size : Number of words for the Bag-of-Words model
            C : C hyperparamter for the model
    """
    def __init__(self,bow_size=200,C=1):
        self.bow_size=bow_size
        self.dict_vectorizer=DictVectorizer()
        self.sift_cluster = KMeans(self.bow_size)
        self.dict_vect = None
        self.C = C
        self.model_number= LogisticRegression(C=self.C)
        self.model_rank = SVC(C=self.C,probability=True,kernel='rbf',decision_function_shape='ovr')
                
    def desc_sift_img_list(self,images):
        """
        This method extracts SIFT features from a list of images
        Input : images : list of images
        Return: imgs_desc : list of SIFT descriptors
        """
        imgs_desc = []
        sift = cv2.xfeatures2d.SIFT_create()
        for image in images:
            _ , desc = sift.detectAndCompute(image, None)
            imgs_desc.append(desc)
        return imgs_desc
        
    def create_bow(self,list_desc):
        """
        Creates a BoW from a list of descrpitors
        Input: list_desc : List of descriptors
        Return: List_bow: list of BoW disctionaries (one dictionary per sample)
        """
        list_bow = []
        for desc in list_desc:
            bow=Counter((self.sift_cluster.predict(desc)))
            list_bow.append(bow)
        return list_bow
    
    def preprocessing(self,list_images):
        """
        Preprocesses images
        Input: list_images: List of images to be preprocessed
        Return: preprocessed_imgs: List of preprocessed images
        """
        preprocessed_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in list_images]
        return preprocessed_imgs
        
    def feature_extraction(self,images,train=False):
        """
        Manages feature extraction by calling methods to extract SIFT descrpitors 
        and to create the Bag-of-Words description. If train mode is True then it also trains a k-means clustering model
        used to cluster descriptors into words (for BoW)
        
        Input: images: list of images
               train: boolean, True if is extracting features from training samples
        Return: Vectorized Bow for each image
        
        """
        features = self.desc_sift_img_list(images)
        
        if train is True:
            concat_im = np.concatenate(features)
            self.sift_cluster.fit(concat_im)
        
        feat_dict = self.create_bow(features)
        if train is True:
            self.dict_vectorizer.fit(feat_dict) #Vectorize is required to work with scikit-learn API
        
        X = self.dict_vectorizer.transform(feat_dict)
        return X
    
    def train(self,list_images,labels):
        """
        Manages training process by calling preprocessing, feature extraction and model fit method.
        Input: list_images: List of training samples
               labels: List of training labels
        Return: self.model_num: Model to classify a card number 
        
        """
        preprocessed_imgs = self.preprocessing(list_images)
        X = self.feature_extraction(preprocessed_imgs,train=True)
        self.model_number.fit(X,labels)        
        return self.model_number
    
    def predict(self,list_images):
        """
        Predicts the number of a playing card
        Input: list_images: List of images to classify
        Return: prob_preds_number: list with model output for each class  (matrix without class names)
                preds_number: list with most probable class
        """
        preprocessed_imgs = self.preprocessing(list_images)
        X = self.feature_extraction(preprocessed_imgs,train=False)
        prob_preds_number = self.model_number.decision_function(X)
        preds_number = self.model_number.predict(X)
        return prob_preds_number,preds_number
    
    def predict_proba(self,list_images,threshold=0):
        """
         Similar to predict but returns a list of model output for each class  that is above a given threshold
         Input: list_images: images to be classified
                threshold: confidence value above which a class is assigned to a sample
                
         Return: list with model output for each class that is above a threshold
        """
        preds_number,_ = self.predict(list_images)
        preds_number.argmax(axis=1)
        return self.model_number.classes_[((preds_number > threshold) * preds_number).nonzero()[1]],preds_number[((preds_number > threshold) * preds_number).nonzero()]
    
    def show_predictions(self,list_images,image_names):
        """
        Builds an interpretable table of predict method output
        """
        preds_number,_ = self.predict(list_images)
        preds_number_pd = pd.DataFrame(preds_number,index=image_names)
        preds_number_pd.columns = self.model_number.classes_        
        return preds_number_pd


# # Pyramid and Sliding window classifier


def pyramid(image, scale=1.5, minSize=(30, 30)):
    """
    Build pyramid images from an orginal sample
    Input: scale: Factor by which previous image's dimensions are decreased
           minSize: Size of the smallest pyramid desired
    Return: pyramid_images: Set of pyramid images
    """
    # yield the original image
    pyramid_images=[]
    #yield image
    pyramid_images.append(image)
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            #break
             return pyramid_images   
            # yield the next image in the pyramid
        pyramid_images.append(image)

def sliding_window(image, stepSize, windowSize):
    """
    Slides a window across an image. 
    Input: image: image from which windows are created
           stepSize: distance between the initial coordinates of consecutive windows
           windowSize: size of each window
    Return: List of top left corner coordinates for each window and window values
    """
    # slide a window across the image
    sliding_windows=[]
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            sliding_windows.append((x, y, image[y:y + windowSize[1], x:x + windowSize[0]]))
    return sliding_windows

def slide(image,cm,winW, winH,threshold=0):
    """
    Calls the classification model for each window of each image pyramide.
    Input: image: image to be classified
           cm: a trained instance of cards model
           threshold: threshold value for predict_proba method
    Return: dict_results: Dictionary with highest confidence for each class that is above threshold
    """
    results = []
    dict_results = {}
    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            sift = cv2.xfeatures2d.SIFT_create()
            _ , desc = sift.detectAndCompute(window, None)
            if desc is None:
                continue
            result = cm.predict_proba([window],threshold)

            
            if len(result[0])>0:
                    results.append(result)
                   

    for (i,j) in results:
        try:
            if dict_results.get(i[0]) < float(j):
                dict_results[i[0]]=float(j)
        except:
            dict_results[i[0]]=float(j)
    return dict_results

def classification(images,templates,true_labels,winW=64,winH=64,threshold=0):
    """
    Classification main function. Creates model and sequentially tries to predict an image number.
    Input: images:list of images to be predicted
           templates: training samples (bouding boxes)
           true_labels: training samples' labels
    Return: list of dictionaries with predicitions for each sample
    """
    pred_labels=[]
    cm = cards_model()
    cm.train(templates,true_labels)
    for curr_img in images:
        result = slide(curr_img,cm,winW,winH,threshold)
        pred_labels.append(result)
    return pred_labels







