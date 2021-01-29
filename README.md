(41, 3)
          
rating  text
category
negativo      26    26
positivo      15    15
Tamanho do dataset de treino: 32
Tamanho do dataset de teste: 9
(32, 100)
(9, 100)

Fitting 10 folds for each of 900 candidates, totalling 9000 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.

Melhores hiper parametros:
{'kernel': 'rbf', 'C': 1, 'gamma': 1, 'degree': 1, 'probability': True}

Acuracia:
0.78125

Acuracia: 1.0
report de acuracia:
              precision    recall  f1-score   support

    negativo       1.00      1.00      1.00         7
    positivo       1.00      1.00      1.00         2

   micro avg       1.00      1.00      1.00         9
   macro avg       1.00      1.00      1.00         9
weighted avg       1.00      1.00      1.00         9

matriz de confusao:
[[2 0]
 [0 7]]
Sem stemmer
C=0.3
Acuracia: 0.777777777778
/Users/pedro.correia/Library/Python/2.7/lib/python/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

    negativo       0.78      1.00      0.88         7
    positivo       0.00      0.00      0.00         2

   micro avg       0.78      0.78      0.78         9
   macro avg       0.39      0.50      0.44         9
weighted avg       0.60      0.78      0.68         9

matriz de confusao:
[[0 2]
 [0 7]]

C=1.0
Acuracia: 1.0
              precision    recall  f1-score   support

    negativo       1.00      1.00      1.00         7
    positivo       1.00      1.00      1.00         2

   micro avg       1.00      1.00      1.00         9
   macro avg       1.00      1.00      1.00         9
weighted avg       1.00      1.00      1.00         9

matriz de confusao:
[[2 0]
 [0 7]]
Com stemmer
C=0.3
Acuracia: 0.777777777778
              precision    recall  f1-score   support

    negativo       0.78      1.00      0.88         7
    positivo       0.00      0.00      0.00         2

   micro avg       0.78      0.78      0.78         9
   macro avg       0.39      0.50      0.44         9
weighted avg       0.60      0.78      0.68         9

matriz de confusao:
[[0 2]
 [0 7]]

C=1.0
Acuracia: 1.0
              precision    recall  f1-score   support

    negativo       1.00      1.00      1.00         7
    positivo       1.00      1.00      1.00         2

   micro avg       1.00      1.00      1.00         9
   macro avg       1.00      1.00      1.00         9
weighted avg       1.00      1.00      1.00         9

matriz de confusao:
[[2 0]
 [0 7]]
