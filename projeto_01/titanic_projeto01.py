
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import  seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import math
import sklearn

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from scipy.stats import randint

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

# In[]:
# ignoirar warnming
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# %matplotlib inline
# import numpy as np
# import pandas as pd
# 
# # Visualisation
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import seaborn as sns
# from __future__ import print_function
# 
# 
# from sklearn.model_selection import train_test_split
# from sklearn import tree

# In[2]:

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


# In[3]:

#Adaptado de https://www.kaggle.com/sachinkulkarni/titanic/an-interactive-data-science-tutorial
#carrega a base de dados
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
full_data = [train, test]



# In[4]:

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


# In[5]:

#Cria a base de treinamento para o projeto (70% das amostras)
full = train.append(test , ignore_index = True)
titanic = full[: train.shape[0]]
print("Datasets:\nCompleto: " , full.shape, "\nTreinamento:", titanic.shape)


# In[6]:

#Imprime as primeiras amostras, juntamento com o cabeçalho
titanic.head()



# In[7]:

#Análise dos dados. Observe que é possível identificar dados inconsistentes. Por exemplo, idade mínima de 0.42!
titanic.describe()


# In[8]:

#Correlação entre as características.
#Pode dar uma ideia do que está relacionado com o que.
plot_correlation_map(titanic)


# In[9]:

#Distribuição das amostras dentro de uma mesma classe
#Visualize a "Survival Rate" em relação aos seguintes atributos: Embarked, Sex, Pclass, SibSp, Parch
plot_categories(titanic, cat = 'SibSp', target = 'Survived')


# In[10]:

#A PARTIR DESTE PONTO SÃO CARREGADOS E PROCESSADOS OS ATRIBUTOS


# In[11]:

#Altera o atributo "Sex" de valores nominais (Male/Female)para 0 e 1
sex = pd.Series(np.where(full.Sex=='male', 1, 0), name = 'Sex')


# In[12]:
'''
#Cria uma nova variável para cada valor único de "Embarked" (no caso, Embarked_C  Embarked_Q  Embarked_S)
embarked = pd.get_dummies(full.Embarked, prefix='Embarked')
print(embarked.head())

'''
#dadosEmbarked
#for row in full:
#    dadosEmbarked += row['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#dadosEmbarked

embarked = full['Embarked'].map( {np.nan:0, 'S': 0, 'C': 1, 'Q': 2} )
embarked.astype(int)
embarked.unique()
#embarked
#Cria uma nova variável para cada valor único de "Embarked" (no caso, Embarked_C  Embarked_Q  Embarked_S)
#embarked = pd.get_dummies(full.Embarked, prefix='Embarked')




#In[]:
#Cria uma nova variável para cada valor único de "Pclass"
pclass = pd.get_dummies(full.Pclass , prefix='Pclass' )
print(pclass.head())

#In[]
'''
#Muitos algoritmos requerem que todas as amostras possuam valores atribuídos para todas as características. 
#No caso de dados faltantes, uma possibilidade é preenchê-los com o valor médio das demais observações.

#Cria o dataset
imputed = pd.DataFrame()

#Preenche os valores que faltam em "Age" com a média das demais idades
imputed['Age'] = full.Age.fillna(full.Age.mean())

#O mesmo para "Fare"
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())

imputed.head()
'''

# In[]
# Age em categorias

#Cria o dataset
imputed = pd.DataFrame()
age = pd.DataFrame()
#Preenche os valores que faltam em "Age" com a média das demais idades
imputed['Age'] = full.Age.fillna(full.Age.mean())
    
imputed.loc[ imputed['Age'] <= 12, 'Age'] = 0
imputed.loc[(imputed['Age'] > 12) & (imputed['Age'] <= 19), 'Age'] = 1
imputed.loc[(imputed['Age'] > 19) & (imputed['Age'] <= 60), 'Age'] = 2
# imputed.loc[(imputed['Age'] > 48) & (imputed['Age'] <= 64), 'Age'] = 3
imputed.loc[ imputed['Age'] > 60, 'Age'] = 3

#Cria nova características para representar o tipo de família 
# imputed['Age'] = age['Age'].map(lambda s : 1 if s <= 12, s : 2 if 12 < s <=19 s : 3 if 19 < s <= 60, s : 4 if 60 < s)
# imputed['Age_adulto']  = age['Age'] .map(lambda s : 1 if 16 <= s < 32 else 0)
# imputed['Age_velho']  = age['Age'].map(lambda s : 1 if 32 <= s <= 50 else 0)
# imputed['Age_Acident']  = age['Age'].map(lambda s : 1 if  50 < s else 0)



#O mesmo para "Fare"
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())
imputed.loc[ imputed['Fare'] <= 7.91, 'Fare'] = 0
imputed.loc[(imputed['Fare'] > 7.91) & (imputed['Fare'] <= 14.454), 'Fare'] = 1
imputed.loc[(imputed['Fare'] > 14.454) & (imputed['Fare'] <= 31), 'Fare'] = 2
imputed.loc[ imputed['Fare'] > 31, 'Fare'] = 3
imputed['Fare'] = imputed['Fare'].astype(int)
imputed.head()



# In[14]:

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
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

#Faz o mapeamento de cada título
title['Title'] = title.Title.map(Title_Dictionary)
#Cria uma nova variável para cada título
title = pd.get_dummies(title.Title)

title.head()


# In[15]:

#Extrai a categoria da cabine a partir do número
cabin = pd.DataFrame()

#Substitui dados faltantes por "U" (Uknown)
cabin['Cabin'] = full.Cabin.fillna( 'U' )

#Mapeia cada valor de cabine com a letra
cabin['Cabin'] = cabin['Cabin'].map(lambda c : c[0])

#Cria uma variável para cada categoria
cabin = pd.get_dummies(cabin['Cabin'] , prefix = 'Cabin')

cabin.head()


# In[16]:

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


# In[17]:

#Cria variáveis para representar o tamanho da família e também cada categoria
family = pd.DataFrame()

#Cria nova característica que representa o tamanho da família (quantidade de membros)
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

#Cria nova características para representar o tipo de família 
family['Family_Single'] = family['FamilySize'].map(lambda s : 1 if s == 1 else 0)
family['Family_Small']  = family['FamilySize'].map(lambda s : 1 if 2 <= s <= 4 else 0)
family['Family_Large']  = family['FamilySize'].map(lambda s : 1 if 5 <= s else 0)

family.head()


# In[18]:

#Seleciona as características que serão incluídas no descritor (vetor de características)
full_X = pd.concat([imputed, embarked, family, sex, title, pclass] , axis=1)
full_X.head()


# In[]:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy
def testarClasficador(clf,DadosX,DadosY):
    valores = cross_val_score(clf, DadosX, DadosY, cv = 5)
    mediaAcc = numpy.mean(valores)
    print("A Media do clf foi: ", mediaAcc)
    print("A Desvio padrao foi: ", numpy.std(valores))
    print("A Variancia foi: ", numpy.var(valores))
    print("Os teste resultaram em: ", valores)
    return mediaAcc

# In[]:
#func para randmonized search
def buscarParametrosRand(clf,DadosX,DadosY,paramatros,inter,nomeClassificador):
    randCLF =  RandomizedSearchCV(clf, paramatros, n_iter = inter)
    randCLF.fit(DadosX,DadosY)
    print("Melhores parametros encontrados %s para: %s" % (nomeClassificador, randCLF.best_params_))
    return randCLF

def buscarParametrosGrid(clf,DadosX,DadosY,paramatros,nomeClassificador):
    randCLF =  GridSearchCV(clf, paramatros)
    randCLF.fit(DadosX,DadosY)
    print("Melhores parametros encontrados %s para: %s" % (nomeClassificador, randCLF.best_params_))
    return randCLF



# In[19]:

#A PARTIR DAQUI, COMEÇA O PROCESSO DE CLASSIFICAÇÃO!

ResultadosClassficadores = {}
#A partir apenas das amostras do arquivo train.csv, cria a base de treinamento e teste.
X = full_X[0:train.shape[0]]
y = titanic.Survived
X_train, X_test, y_train, y_test = train_test_split(X , y, train_size = .75)

# Arvore de decisão

clfTree = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
clfTree = clfTree.fit(X_train, y_train)
preditor = clfTree.predict(X_test)



acc = sklearn.metrics.accuracy_score(y_test, preditor)
acc


a = tree.plot_tree(clfTree, feature_names=X.columns.values, class_names=['Dead', 'Survived'], max_depth = 3)

# print da arvore
'''
out = StringIO()
tree.export_graphviz(clf, out_file=out, feature_names=X.columns.values, class_names=y.name, filled=True, rounded=True, special_characters=True)
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
with open('irisDT-RS.png', 'wb') as f:
    f.write(graph.create_png())    
Image("irisDT-RS.png") 

'''



# In[ ]:
# Knn
clf_KNN = KNeighborsClassifier()
paramatros = {"n_neighbors": range(2,10), "weights": ['uniform','distance']}


randKnn =  buscarParametrosGrid(clf_KNN,X,y,paramatros,"KNN")
clf_KNN = randKnn.best_estimator_

# testar
resultadoKNN= testarClasficador(clf_KNN, X,y)
ResultadosClassficadores["resultadoKNN"] = resultadoKNN

# In[ ]:

# Random Forests
# random forests sao conjuntos de arvores de decisao com o objetivo de reduzir o overfitting do modelo anterior

from sklearn.ensemble import RandomForestClassifier
print("Teste Random Forest")
clfRF = RandomForestClassifier()

# Procurar por parametros otimos
param_dist = {"criterion":["gini", "entropy"],
             "min_samples_split": randint(6, 10),
             "max_depth": randint (8, 10),
             "min_samples_leaf":randint(2, 6),
             "max_leaf_nodes":randint(6, 8)}


randRF =  RandomizedSearchCV(clfRF, param_dist, n_iter = 100)
randRF.fit(X,y)

print("Melhores parametros encontrados para Random florrests: ", randRF.best_params_)
clfRF = randRF.best_estimator_
# testar
resultadoRandomForets = testarClasficador(clfRF, X,y)
ResultadosClassficadores["randomForests"] = resultadoRandomForets




# In[]:
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
print("Teste naive bayes")

gnb = GaussianNB()

resultadoGNB = testarClasficador(gnb,X,y)
gnb.fit(X,y)
ResultadosClassficadores["naiveBayes"] = resultadoGNB


# o naive bayes tem apenas um hyper parametro
# este parametro eh o var_smoothing
# vou pesquisar como melhorar ele

#%%
# In[]:
# Resumo dos resultados
ResultadosClassficadores

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
gerarSubmicoes(clfTree, testX, passagemId,'tree-predict.csv')

#submissao Knn
gerarSubmicoes(clf_KNN, testX, passagemId,'knn-predict.csv')

# submissao do Random forets
gerarSubmicoes(clfRF, testX, passagemId,'rf-predict.csv')

# submissao do Naive bayes
gerarSubmicoes(gnb, testX, passagemId,'gnb-predict.csv')


# In[]
#PCA
from sklearn.decomposition import PCA
for i in range(5,10):
    print("testando reducao com %s features",i)
    pca = PCA(n_components=i)
    pca.fit(X)  

    pca2 = PCA(n_components=i)
    pca2.fit(testX)  


    gnb = GaussianNB()
    x1 = pca.transform(X)
    gnb.fit(x1,y)


    print(pca.explained_variance_ratio_)  

    print(pca.singular_values_)  
    testarClasficador(gnb,x1,y)



#%%
#In:[]
# Procurando melhores features
from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd

clfKBest = SelectKBest(score_func=chi2, k='all')
clfKBest.fit(X, y)

featureSelect = pd.DataFrame(clfKBest.scores_)
featureSelect

dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,featureSelect],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores.sort_values(by=['Score'])


classKbest = SelectKBest(score_func=chi2, k=10)
classKbest.fit(X,y)
X_new = classKbest.transform(X)

X_teste_new = classKbest.transform(testX)

print("Teste Random Forest parametros cortados")
clfRF2 = RandomForestClassifier()

# Procurar por parametros otimos
param_dist = {"criterion":["gini", "entropy"],
             "min_samples_split": randint(6, 10),
             "max_depth": randint (8, 10),
             "min_samples_leaf":randint(2, 6),
             "max_leaf_nodes":randint(6, 8)}


randRF2 =  RandomizedSearchCV(clfRF2, param_dist, n_iter = 100)
randRF2.fit(X_new,y)

print("Melhores parametros encontrados para Random florrests: ", randRF.best_params_)
clfRF2 = randRF2.best_estimator_
# testar
resultadoRandomForets = testarClasficador(clfRF2, X_new,y)
ResultadosClassficadores["randomForestsParametrosCortados"] = resultadoRandomForets

gerarSubmicoes(clfRF2, X_teste_new, passagemId,'rf-predict-corte.csv')
featureScores

#%%
