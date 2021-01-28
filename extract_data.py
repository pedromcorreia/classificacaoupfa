import requests
import json
import csv
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB


TOKEN = ''

def resquest_upas(search):
    dic = {}
    url = 'https://maps.googleapis.com/maps/api/place/textsearch/json?query=' + search + '&key=' + TOKEN
    req = requests.get(url)

    json_obj = json.loads(req.content)

    for result in json_obj['results']:
        dic[result['name']] = result['place_id']
    return dic

def resquest_upas_mock(_search):
    return {u'Unidade de Pronto Atendimento Campo Comprido': u'ChIJ5VZRmyTi3JQRdmel94mPJMc', u'Health Unit Santa Candida': u'ChIJMbjnAkDm3JQR7QLLTagnxuQ', u'UPA Boqueir\xe3o': u'ChIJ3X6qFyf73JQRqGuRVv4QdDE', u'UPA Pinheirinho': u'ChIJJ38-Or383JQRdkGFtPKJaGA', u'UPA Campo Comprido': u'ChIJa2Ra0nzj3JQRRPqZZyo7XV0', u'Health Unit Campina do Siqueira': u'ChIJRYCy2cPj3JQR7SqKpKJRWbA', u'UPA 24 Boa Vista': u'ChIJ5ez1b2_m3JQRdh6MFffUkhs', u'UPA CIC': u'ChIJF68ZJezj3JQRUwDRBf3Pf3A', u'Santa Casa de Curitiba Emergency Care': u'ChIJteqnPm7k3JQRRtahmQR6Mp0', u'Health Unit Ouvidor Pardinho': u'ChIJf-AAmYvk3JQRuUfnFNJNSc4', u'UPA Boa Vista': u'ChIJdy1LRnzn3JQR6J0ordck06k', u'UPA Fazendinha': u'ChIJFYExsozj3JQRxSeNW2jerEU', u'UPA Cajuru': u'ChIJb12CpYP73JQRYofI31F81HQ', u'UPA Cajuru - Unidade de Pronto Atendimento 24h': u'ChIJBR5Bfav63JQRNllWF1VACUQ', u'UPA 24 Pinhais.': u'ChIJfxX0H5fv3JQRBYOLKp_o-oY', u'UPA S\xedtio Cercado': u'ChIJbUaalJ773JQR_HgBpKPA2H0', u'Unidade de Sa\xfade Abranches': u'ChIJE4Gtr9rm3JQR59_I7pcgusQ', u'UPA Tatuquara': u'ChIJzQE2q8_93JQRmwSH7_lYH-g'}

def filter_results():

    dic = resquest_upas_mock('upa+curitiba')
    newDict = dict()
    for (key, value) in dic.items():
       if  'UPA' in key:
           newDict[key] = value
    return newDict

def get_review(place_id):
    url =  'https://maps.googleapis.com/maps/api/place/details/json?placeid=' + place_id + '&key=' + TOKEN
    req = requests.get(url)
    json_obj = json.loads(req.content)

    try:
        for review in json_obj['result']['reviews']:
            ndic = {}
            ndic['rating'] = review['rating']
            ndic['text'] = review['text']
            #write_values(ndic)
            write_csv(ndic)
    except Exception as e:
        pass


def write_values(my_dict):
    with open("ratings.txt", "a") as f:
        rating = ">classe: " + str(my_dict["rating"])
        text = my_dict["text"].encode("UTF-8")
        f.write(rating)
        f.write('\n')
        f.write(text)
        f.write('\n')

def write_csv(my_dict):
    filename = "records.csv"
    row = [str(my_dict["rating"]), my_dict["text"].encode("UTF-8")]
    print row
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)
def init_csv():
    fields = ['rating', 'text']
    filename = "records.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)


def get_reviews():
    search = {}
    search = filter_results()

    init_csv()

    for (key, value) in search.items():
        get_review(value)

def vectorize_texts(text, translate):
    word_vector = [0] * len(translate)
    for word in text:
            if word in translate:
                index = translate[word]
                word_vector[index] += 1
    return word_vector

def read_pandas():
    reviews = pd.read_csv('records.csv')
    rating = reviews['rating']
    comments = reviews['text']
    comments_split = comments.str.lower().str.split()
    words = set()
    for word in comments_split:
        words.update(word)

    indexes = range(len(words))
    translate = {word: index for word, index in zip(words, indexes)}
    vectorize_text = [vectorize_texts(text, translate) for text in comments_split]
    model = MultinomialNB()
    modelGaussianNB = GaussianNB()
    model.fit(vectorize_text, rating)
    modelGaussianNB.fit(vectorize_text, rating)
    predicted = model.predict(vectorize_text)
    predictedGaussianNB = modelGaussianNB.predict(vectorize_text)
    print (vectorize_text)
    print(predicted)
    print(predictedGaussianNB)


get_reviews()
read_pandas()
