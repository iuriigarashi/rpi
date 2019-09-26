# In[]:
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from __future__ import print_function

import math

get_ipython().magic(u'matplotlib inline') #para imprimir no próprio notebook

# In[]:
# Importing SKLearn clssifiers and libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.naive_bayes import GaussianNB


# In[]:
#Funções auxiliares
def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots(figsize = (12,10))
    cmap = sns.diverging_palette(220,10, as_cmap=True)
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={'shrink':.9}, 
        ax=ax, 
        annot=True, 
        annot_kws={'fontsize':12}
    )
    
def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df,row = row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

# In[]:
#carrega a base de dados
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
full_data = [train, test]

# In[]:
#Identifica as características presentes
train.columns.values

#Descrição das variáveis
#We've got a sense of our variables, their class type, and the first few observations of each. We know we're working with 1309 observations of 12 variables. To make things a bit more explicit since a couple of the variable names aren't 100% illuminating, here's what we've got to deal with:
#Variable Description
#Survived: Survived (1) or died (0)
#Pclass: Passenger's class
#Name: Passenger's name
#Sex: Passenger's sex
#Age: Passenger's age
#SibSp: Number of siblings/spouses aboard
#Parch: Number of parents/children aboard
#Ticket: Ticket number
#Fare: Fare
#Cabin: Cabin
#Embarked: Port of embarkation

# In[]:
#Cria a base de treinamento para o projeto (70% das amostras)
full = train.append(test , ignore_index = True)
titanic = full[: train.shape[0]]
print("Datasets:\nCompleto: " , full.shape, "\nTreinamento:", titanic.shape)

# In[]:
#Análise dos dados. Observe que é possível identificar dados inconsistentes. Por exemplo, idade mínima de 0.42!
titanic.describe()


# In[]:
#Correlação entre as características.
#Pode dar uma ideia do que está relacionado com o que.
plot_correlation_map(titanic)
# In[]:
#Distribuição das amostras dentro de uma mesma classe
#Visualize a "Survival Rate" em relação aos seguintes atributos: Embarked, Sex, Pclass, SibSp, Parch
plot_categories(titanic, cat = 'SibSp', target = 'Survived')
# In[]:
# Tratamento dos dados
#Altera o atributo "Sex" de valores nominais (Male/Female)para 0 e 1
sex = pd.Series(np.where(full.Sex=='male', 1, 0), name = 'Sex')

# In[]:
# Cria embarked numérico. S = 0, C = 1, Q = 2, nan = random(0-2)
import random
Embarked_Dictionary = {
                    np.nan:       random.randint(-1,2),
                    "S":        0,
                    "C":      1,
                    "Q":   2,
                    }

    
embarked = full['Embarked'].map(Embarked_Dictionary)
print(embarked.head())
#Cria uma nova variável para cada valor único de "Pclass"
pclass = pd.get_dummies(full.Pclass , prefix='Pclass' )
print(pclass.head())
# In[]:
#As distinções refletiam o status social e podem ser utilziados para prever a probabilidade de sobrevivência

title = pd.DataFrame()

#Extrai o título de cada nome
title['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

#Lista agregada de títulos
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Miss",
                    "Mlle":       "Miss",
                    "Ms":         "Miss",
                    "Mr" :        "Mr",
                    "Mrs" :       "Miss",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

#Faz o mapeamento de cada título
title['Title'] = title.Title.map(Title_Dictionary)
#Cria uma nova variável para cada título
title = pd.get_dummies(title.Title)

title.head()
# In[]:
#Extrai a categoria da cabine a partir do número
cabin = pd.DataFrame()

#Substitui dados faltantes por "U" (Uknown)
cabin['Cabin'] = full.Cabin.fillna( 'U' )

#Mapeia cada valor de cabine com a letra
cabin['Cabin'] = cabin['Cabin'].map(lambda c : c[0])

#Cria uma variável para cada categoria
cabin = pd.get_dummies(cabin['Cabin'] , prefix = 'Cabin')

cabin.head()
# In[]:
#Extrai a classe de cada ticket a partir do seu número
#Caso não tenha prefixo, retorna XXX
def cleanTicket( ticket ):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

#Cria uma nova variável para cada caso
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape
ticket.head()
# In[]:
#Cria variáveis para representar o tamanho da família e também cada categoria
family = pd.DataFrame()

#Cria nova característica que representa o tamanho da família (quantidade de membros)
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

print(family)

#Cria nova características para representar o tipo de família 
family['Family_Single'] = family['FamilySize'].map(lambda s : 1 if s == 1 else 0)
family['Family_Small']  = family['FamilySize'].map(lambda s : 1 if 2 <= s <= 4 else 0)
family['Family_Large']  = family['FamilySize'].map(lambda s : 1 if 5 <= s else 0)

family.head()
# In[]:

imputed = pd.DataFrame()
age = pd.DataFrame()
#Preenche os valores que faltam em "Age" com a média das demais idades
imputed['Age'] = full.Age.fillna(full.Age.mean())

imputed.loc[ imputed['Age'] <= 12, 'Age'] = 0
imputed.loc[(imputed['Age'] > 12) & (imputed['Age'] <= 19), 'Age'] = 1
imputed.loc[(imputed['Age'] > 19) & (imputed['Age'] <= 60), 'Age'] = 2
imputed.loc[ imputed['Age'] > 60, 'Age'] = 3
#O mesmo para fare
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())
imputed.loc[ imputed['Fare'] <= 7.91, 'Fare'] = 0
imputed.loc[(imputed['Fare'] > 7.91) & (imputed['Fare'] <= 14.454), 'Fare'] = 1
imputed.loc[(imputed['Fare'] > 14.454) & (imputed['Fare'] <= 31), 'Fare'] = 2
imputed.loc[ imputed['Fare'] > 31, 'Fare'] = 3
imputed['Fare'] = imputed['Fare']
imputed.head()
# In[]:

#Seleciona as características que serão incluídas no descritor (vetor de características)
full_X = pd.concat([imputed, embarked, family, sex, title, pclass] , axis=1)
full_X.head()
# In[]:
#A PARTIR DAQUI, COMEÇA O PROCESSO DE CLASSIFICAÇÃO!

#A partir apenas das amostras do arquivo train.csv, cria a base de treinamento e teste.
X = full_X[0:train.shape[0]]
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X , y, train_size = .8)

# In[]:
# Funções Auxiliares ficarão nesta célula
# Função de testar um classificador
# Entrada: classificador, X, y, e cv_number
# Saída: print do valor médio do cross validation
import numpy as np
def testar_classificador(clf, data_X, data_Y, cv_number):
    v = cross_val_score(clf, data_X, data_Y, cv = cv_number)
    media = np.mean(v)
    print("Média por Cross Validation: " + str(media))

# In[]:
# Árvores de Decisão
# Parametros
#parametros = {
#            "criterion": ['gini', 'entropy'],
#            "max_depth": range(2,10),
#            }
clf_T = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 4)
clf_T = clf_T.fit(X_train, y_train)
tree_preditor = clf_T.predict(X_test)
# Testando para cv = 5
print("Árvores de Decisão:\n")
testar_classificador(clf_T, X, y, 5)

# In[]:
# KNN
clf_KNN = KNeighborsClassifier(n_neighbors = 2, weights = 'distance')
# Parametros
#parametros = {
#            "n_neighbors": range(2,10), 
#            "weights": ['uniform','distance'],
#            "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute'],
#            "p" : range(1,5)
#            }
#clf_KNN = GridSearchCV(clf_KNN, parametros)
clf_KNN.fit(X_train, y_train)
preditor_KNN = clf_KNN.predict(X_test)
# Testando para cv = 5
print("KNN:\n")
testar_classificador(clf_KNN, X, y, 5)

# In[]:
# Random Forest
parametros = {
            "criterion": ['gini', 'entropy'],
            "max_depth": range(2,10),
            }
clf_RF = RandomForestClassifier()
#clf_RF.fit(X_train, y_train)
randCLF =  GridSearchCV(clf_RF, parametros)
randCLF.fit(X_train,y_train)
clf_RF = randCLF.best_estimator_
clf_RF.fit(X_train, y_train)
preditor_RF = clf_RF.predict(X_test)
print(clf_RF.criterion)
print(clf_RF.max_depth)
# Testando para cv = 5
print("Random Forests:\n")
testar_classificador(clf_RF, X, y, 5)

# In[]:
# Naive Bayes
clf_NB = GaussianNB()
clf_NB.fit(X_train, y_train)
preditor_NB = clf_NB.predict(X)
# Testando para cv = 5
print("Naive Bayes:\n")
testar_classificador(clf_NB, X, y, 5)

# In[]:
# Gerar submicoes para a competicao

def gerarSubmicoes(clf,dadosEntrada, id,arqSaida):
    dadosPredicao = clf.predict(dadosEntrada)
    
    dados = zip(id,dadosPredicao)
    #adicioanr o pasagem id

    dadosCSV = pd.DataFrame(dados,columns={'PassengerId','Survived'})
    
    dadosCSV['Survived'] = dadosCSV['Survived'].astype(int)
    dadosCSV.to_csv(arqSaida,index=False,float_format='%.f')


    return dadosCSV

# Recupera os dados de teste para se realizar a predicao
dataBaseFull = pd.DataFrame(full_data)
testX = full_X[train.shape[0]:]
passagemId = test.ix[:,0]

# submissao do Arvore de decisão
gerarSubmicoes(clf_T, testX, passagemId,'tree-predict.csv')

# submissao Knn
gerarSubmicoes(clf_KNN, testX, passagemId,'knn-predict.csv')

# submissao do Random forets
gerarSubmicoes(clf_RF, testX, passagemId,'rf-predict.csv')

# submissao do Naive bayes
gerarSubmicoes(clf_NB, testX, passagemId,'gnb-predict.csv')


#%%
