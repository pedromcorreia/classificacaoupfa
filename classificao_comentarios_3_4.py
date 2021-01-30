#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[3]:


# Leitura do csv com todas as avaliações coletadas
path = "records.csv"
df = pd.read_csv(path)


# In[30]:


# Separação do dataset de treino e teste:

# treino = avaliações com pontuação 1, 2 e 5
train = df[df['rating'].isin([1, 2, 5])]

# teste = avaliações com pontuação 3 e 4
test = df[df['rating'].isin([3, 4])]


# In[33]:


# Quantidade de avaliações em cada dataset
print('Dados de treino:', train.shape[0])
print('Dados de teste:',test.shape[0])


# In[39]:


# Classificação no dataset de treino do que é uma avaliação positiva ou negativa
# Conforme regra é positivo quando a nota é 5 e negativo quando a nota é 1 ou 2.
# Nesse caso, como só temos essas duas opções, setei tudo pra positivo e depois só mudo pra negativo o que é 1 ou 2
train['category'] = 'positivo'
train.loc[train['rating'].between(1, 2, inclusive=True), 'category'] = 'negativo'


# In[40]:


# Validando o resultado pra ter certeza que criei as regras corretamente. 
# Se você somar o rating 1 e 2 vai ver que dá 26, igual a quantidade da categoria negativo
print(train.groupby(['rating']).size())
print('\n\n', train.groupby(['category']).size())


# In[60]:


# Separando o que é dado de treino e de teste (sendo treino tudo 1, 2 e 5 e teste 3 e 4)

x_train = train['text'].values
y_train = train['category'].values
x_test = test['text'].values

# Guardo o rating dos dados de teste porque não temos uma classificação exata para eles, 
# mas talvez o rating nos ajude a entender se o modelo classificou certo ou não
rating_test = test['rating'].values


# In[45]:


print('Tamanho do dataset de treino: {}'.format(len(x_train)))
print('Tamanho do dataset de teste: {}'.format(len(x_test)))


# In[48]:


# Transformo o texto em códigos reconhecidos pelo modelo
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1,2),
                        max_df=0.8,
                        max_features=100)


# In[49]:


# Alimento/treino o modelo com os dados de treino
features_train = tfidf.fit_transform(x_train)
labels_train = y_train

features_test = tfidf.transform(x_test)


# In[50]:


print(features_train.shape)
print(features_test.shape)


# In[55]:


# Envio os parâmetros para o GridSearch definir qual a melhor combinação de parâmetros para meu modelo performar bem

C = [.001, .01, .1, 1, 10]
degree = [1, 2, 3, 4, 5]
gamma = [0.001, 0.01, 0.1, 1, 10, 100]
probability = [True, False]
kernels = ['linear', 'poly', 'rbf']

param_grid = {'C': C, 'kernel':kernels, 'probability':probability, 'gamma':gamma, 'degree':degree}

svc = svm.SVC(random_state=8)

grid_search = GridSearchCV(estimator=svc,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=10,
                           verbose=1)

grid_search.fit(features_train, labels_train)


# In[56]:


# Exibo a melhor combinação de parâmetros e a acurácia obtida com essa combinação
print('Melhores hiper parametros:')
print(grid_search.best_params_)
print('Acuracia:')
print(grid_search.best_score_)


# In[57]:


# Agora treino meu modelo usando os melhoes parâmetros e com os dados de treino
best_model = grid_search.best_estimator_
best_model.fit(features_train, labels_train)


# In[58]:


# Com meu modelo já treinado (ele ja aprendeu com os dados de treino), submeto os dados de teste (rating 3 e 4) 
# para que com base no que ele aprendeu ele possa classificar os comentários que ainda não sabemos se é positivo ou negativo porque a nota não foi 1, 2 ou 5
y_predict = best_model.predict(features_test)


# In[59]:


# O resultado fica sempre na variável de retorno do .predict
print(y_predict)


# In[62]:


# Agora itero por cada uma das avaliações que eu queria que o modelo classificasse
# Exibo a avaliação, qual o rating da avaliação e qual foi a classificação que o modelo predisse.
for i in range(len(x_test)):
    print("Texto: {}\n\nRating do comentário: {}\nClassificação predita: {}\n\n\n".format(x_test[i], rating_test[i], y_predict[i]))

