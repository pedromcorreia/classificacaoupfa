import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file = "preclassificado.csv"
file_test = "classificado.csv"
Labels=['positivo', 'negativo']

d = pd.read_csv(file)
d_test = pd.read_csv(file_test)
category = d['category']
comments = d['text']

df = pd.DataFrame(data=d)
df_test = pd.DataFrame(data=d_test)

#df.head()

#print(df[['category', 'text']].values)
#print(df)
#print(category.count())

#definicao do shape as categorias
print(df.shape)

#total das categorias
print(df.groupby('category').count())

#definimos parametros de teste e treino
x_train, x_test, y_train, y_test = train_test_split(df['text'].values,
                                                    df['category'].values,
                                                    test_size=0.20,
                                                    random_state=10)

print('Tamanho do dataset de treino: {}'.format(len(y_train)))
print('Tamanho do dataset de teste: {}'.format(len(y_test)))

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1,2),
                        max_df=0.8,
                        max_features=100)

features_train = tfidf.fit_transform(x_train)
labels_train = y_train

features_test = tfidf.transform(x_test)
labels_test = y_test

print(features_train.shape)
print(features_test.shape)

#definimos fatores de predicao
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


print('Melhores hiper parametros:')
print(grid_search.best_params_)
print('Acuracia:')
print(grid_search.best_score_)

best_model = grid_search.best_estimator_
best_model.fit(features_train, labels_train)

y_predict = best_model.predict(features_test)
print("Acuracia: {}".format(accuracy_score(labels_test, y_predict)))
print('report de acuracia:')
print(classification_report(labels_test,y_predict))

print('matriz de confusao:')
print(confusion_matrix(labels_test, y_predict, labels=Labels))


print('Sem stemmer')
best_model = svm.SVC(kernel='linear', C=.3, probability=True)
best_model.fit(features_train, labels_train)
y_predict = best_model.predict(features_test)

print('C=0.3')
print("Acuracia: {}".format(accuracy_score(labels_test, y_predict)))
print(classification_report(labels_test,y_predict))
print('matriz de confusao:')
print(confusion_matrix(labels_test, y_predict, labels=Labels))

best_model = svm.SVC(kernel='linear', C=1.0, probability=True)
best_model.fit(features_train, labels_train)
y_predict = best_model.predict(features_test)

print('\nC=1.0')
print("Acuracia: {}".format(accuracy_score(labels_test, y_predict)))
print(classification_report(labels_test,y_predict))
print('matriz de confusao:')
print(confusion_matrix(labels_test, y_predict, labels=Labels))

print('Com stemmer')
best_model = svm.SVC(kernel='linear', C=.3, probability=True)
best_model.fit(features_train, labels_train)
y_predict = best_model.predict(features_test)

print('C=0.3')
print("Acuracia: {}".format(accuracy_score(labels_test, y_predict)))
print(classification_report(labels_test,y_predict))
print('matriz de confusao:')
print(confusion_matrix(labels_test, y_predict, labels=Labels))

best_model = svm.SVC(kernel='linear', C=1.0, probability=True)
best_model.fit(features_train, labels_train)
y_predict = best_model.predict(features_test)

print('\nC=1.0')
print("Acuracia: {}".format(accuracy_score(labels_test, y_predict)))
print(classification_report(labels_test,y_predict))
print('matriz de confusao:')
print(confusion_matrix(labels_test, y_predict, labels=Labels))


x_train = df['text'].values
y_train = df['category'].values
x_test = df_test['text'].values
