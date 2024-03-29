
# coding: utf-8

# In[1]:

# importa os pacotes necessários
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from skimage.feature import hog
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import numpy as np
import os
import cv2
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import pickle


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
from sklearn import preprocessing

# funções de leitura e preparação das imagens
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0:
            print('Processed {} of {}'.format(i, count))
    return data


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    #image = cv2.imread(image_file)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()

def show_results(classifiers, X_train, y_train, descritor):
    for clf in classifiers:
        name = clf.__class__.__name__
        
        clf.fit(X_train, y_train)

        print("="*30)
        print(name)
        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = clf.score(X_test, y_test)
        print("accuracy: {:.2f}%".format(acc * 100))
        adicionarResultado(acc,name,clf,descritor)


tabelaResultado = []

def adicionarResultado(score,nomeClass,desc,descritor):
    tabelaResultado.append([score,nomeClass,desc,descritor])

def salvarResultado(nomearquivo):
    df = pd.DataFrame(tabelaResultado, columns=["score","nome","desc","descritor"])
    df.to_csv(nomearquivo)


TRAIN_DIR = 'kaggle/train/'


ROWS = 128
COLS = 128
CHANNELS = 3
NIM = 25000

# full dataset
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_images = train_images[:NIM]


# considera apenas NIM imagens. Para o dataset completo, desconsiderar.
#random.shuffle(train_images)

# Leitura das imagens
train = prep_data(train_images)
print("Train shape: {}".format(train.shape))

# Cria os labels (rótulos)
train_labels = pd.read_csv('trainLabels.csv')
train_labels = train_labels[:NIM]


le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
labels_encoded=le.fit_transform(train_labels.iloc[:, 1])



# In[ ]:
# Função que determina os Thresholds lower e upper
# do detector de Canny automaticamente
def auto_canny(image):
    sigma = 0.33
    mediana = np.median(image)
    threshold_baixo = int(max(0, (1.0 - sigma) * mediana))
    threshold_alto = int(min(255, (1.0 + sigma) * mediana))
    bordas = cv2.Canny(image, threshold_baixo, threshold_alto)
    return bordas


rawImages = []
descHist = []
descEdges = []
descSobelX = []
descSobelY = []
descOrientation = []
descHOG = []

count = len(train_images)

# In[]
# carrega os dados anteriores?


def pickelObject(objeto, arquivo):
    file = open(arquivo, 'wb')
    pickle.dump(objeto, file)
    file.close()


def despickel(arquivo):
    file = open(arquivo, 'rb')
    objecto = pickle.load(file)
    file.close()
    return objecto


carregarDados = False
if carregarDados:
    # carrega dados
    print("carregar dados em pickel")
    rawImages = despickel('rawImagesPickel')
    print(rawImages[0].shape)
    descHist = despickel('descHistPickel')
    print(descHist[0].shape)
    descEdges = despickel('descEdgesPickel')
    print(descEdges[0].shape)
    descHOG = despickel('descHOGsPickel')
    print(descEdges[0].shape)
    descSobelX = despickel('descSobelXPickel')
    print(descEdges[0].shape)
    descSobelY = despickel('descSobelYPickel')
    print(descEdges[0].shape)
else:
    hog = cv2.HOGDescriptor()
    for i, image_file in enumerate(train_images):
        image = read_image(image_file)  # Lê a imagem
        # Põe a imagem num vetor de características
        pixels = image_to_feature_vector(image)
        histogram = extract_color_histogram(
            image)  # Extrai o histograma da imagem
        # Aplica o canny e põe num vetor de características
        edges = image_to_feature_vector(auto_canny(image))
        sobeldx = image_to_feature_vector(
            cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0))
        sobeldy = image_to_feature_vector(
            cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0))
        hogimage = image_to_feature_vector(hog.compute(image))

        rawImages.append(pixels)
        descHist.append(histogram)
        descEdges.append(edges)
        descSobelX.append(sobeldx)
        descSobelY.append(sobeldy)
        descHOG.append(hogimage)
        if i % 250 == 0:
            print('Processed {} of {}'.format(i, count))
    pickelObject(rawImages, 'rawImagesPickel')
    pickelObject(descHist, 'descHistPickel')
    pickelObject(descEdges, 'descEdgesPickel')
    pickelObject(descSobelX, 'descSobelXPickel')
    pickelObject(descSobelY, 'descSobelYPickel')
    pickelObject(descHOG, 'descHOGsPickel')

def listsClass(clf):
    return (clf.__class__.__name__, clf)

def criarListaTuplas(lista):
    listaNova = []
    for l in lista:
        listaNova.append(listsClass(l))
    return listaNova

classifiers = [
    GaussianNB(),
     MLPClassifier(activation='relu', solver='adam', alpha=1e-4,
                  hidden_layer_sizes=np.full(50,20),max_iter=500),
    MLPClassifier(activation='identity', solver='adam', alpha=1e-4,
                  hidden_layer_sizes=np.full(50,20),max_iter=500),
    MLPClassifier(activation='relu', solver='adam', alpha=1e-4,
                  hidden_layer_sizes=np.full(100,20),max_iter=500),
                  MLPClassifier(activation='identity', solver='adam', alpha=1e-4,
                  hidden_layer_sizes=np.full(100,100),max_iter=500) 
]
from sklearn.ensemble import  VotingClassifier
#votingClass = VotingClassifier(estimators=criarListaTuplas(classifiers),voting="hard")
#classifiers.append(votingClass)

# In[ ]:

# Avalia o primeiro descritor: as imagens raw
print('\n\ndescritor imagens raw')
(X_train, X_test, y_train, y_test) = train_test_split(
    rawImages, labels_encoded, test_size=0.25, random_state=42)
'''
clf =  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=11, p=2,
                         weights='uniform')

clf.fit(X_train, y_train)
'''

show_results(classifiers, X_train, y_train,'RAW')
#show_results([votingClass], X_train, y_train)

# In[ ]:

# Avalia o quarto descritor = Sobel X
print("\n\nsobel dx")
(X_train, X_test, y_train, y_test) = train_test_split(
    descSobelX, labels_encoded, test_size=0.25, random_state=42)

show_results(classifiers, X_train, y_train,'sobel dx')

# In[ ]:

# Avalia o quinto descritor = Sobel Y
print("\n\nsobel dy")
(X_train, X_test, y_train, y_test) = train_test_split(
    descSobelY, labels_encoded, test_size=0.25, random_state=42)

show_results(classifiers, X_train, y_train,'sobel dy')

# In[ ]:

# Avalia o segundo descritor: color histogram
print('\n\nhistograma de cor')
(X_train, X_test, y_train, y_test) = train_test_split(
    descHist, labels_encoded, test_size=0.25, random_state=42)

show_results(classifiers, X_train, y_train,'histograma')

# In[ ]:
print('\n\ncanny')
(X_train, X_test, y_train, y_test) = train_test_split(
    descEdges, labels_encoded, test_size=0.25, random_state=42)

show_results(classifiers, X_train, y_train,'canny')

# In[ ]:
print('\n\nhog')
(X_train, X_test, y_train, y_test) = train_test_split(
    descHOG, labels_encoded, test_size=0.25, random_state=42)

show_results(classifiers, X_train, y_train,'hog')

# In[ ]:
'''
# Avalia a combinação de todos os descritores
print("\n\nteste com todos os descritores")
trainAux = np.hstack((descHist, descEdges, descSobelX, descSobelY, descHOG))
(X_train, X_test, y_train, y_test) = train_test_split(
    trainAux, labels_encoded, test_size=0.25, random_state=42)

show_results(classifiers, X_train, y_train)
'''

# In[ ]:
salvarResultado('cipher1030k-MLPProfundasENaive.csv')
