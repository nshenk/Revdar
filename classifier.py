import xgboost, textblob, string, nltk, re, geopy, sklearn, requests, collections
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from sklearn import decomposition, ensemble
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from geotext import GeoText
from geopy.geocoders import Nominatim
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

#takes large amounts of text and extracts only the location names;
#creates two dictionaries, one where the values are a frequency count,
#and another where the values are a bunch of relevant headlines
class locations :

    #retrieve data and do some preprocessing
    def filter_data(self) :
        #open file
        with open('redditData.csv', 'r', encoding="utf-8") as f :
            data = f.read()
        #remove numbers
        data = re.sub(r'\d+', '', data)
        #remove special characters
        data = re.sub('[^A-Za-z0-9]+', ' ', data)
        #remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(data)
        data_filtered = [i for i in tokens if not i in stop_words]
        data_filtered = []
        for i in tokens :
            if not i in stop_words :
                data_filtered.append(i)
        return data_filtered

    #use nltk to extract location names from text and filter out irrelevant ones
    def extract_locs(self, data) :
        locations = []
        locs = []
        for item in data :
            locations.append(GeoText(item))
        for i in locations :
            locs.append(i.countries)
        return [', '.join(locations) for locations in locs]

    #get rid of empty values
    def further_filter(self, data) :
        filtered_data = []
        for item in data :
            location = item
            if len(location) > 0 :
                filtered_data.append(location)
        return filtered_data

    #returns a dictionary with frequency count of each location
    def freq_count(self, data) :
        freq = nltk.FreqDist(data)
        freq_dict = {}
        for key,val in freq.items() :
            freq_dict[key] = val
        return freq_dict

    #uses the news api to return recent headlines about country + protest in a dictionary
    def get_article(self, data) :
        article_dict = {}
        list = []
        for country in data :
            url = "https://newsapi.org/v2/everything?q=" + country + " protest&from=2019-11-11&sortBy=relevancy&apiKey=cb15de41dabd4c7d93082e83f477a859"
            response = requests.get(url).json()
            article = response["articles"]
            for ar in article :
                list.append(ar["title"])
            article_dict[country] = list
            list = []
        return article_dict

#trains an algorithm to classify protests as violent or peaceful based on recent related headlines
class violence_classify :

    #preprocess the headlines for classification and return just sorted values
    def clean_data(self, data):
        clean_dict = {}
        news_list = []
        for country, news in data.items() :
            for headline in news :
                #remove numbers
                headline = re.sub(r'\d+', '', headline)
                #remove special characters
                headline = re.sub('[^A-Za-z0-9]+', ' ', headline)
                #remove stop words
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(headline)
                data_filtered = [i for i in tokens if not i in stop_words]
                data_filtered = []
                for i in tokens :
                    if not i in stop_words :
                        data_filtered.append(i)
                #lemmatization
                lemmatizer = WordNetLemmatizer()
                for i in data_filtered :
                    i = lemmatizer.lemmatize(i)
                news_list.append(data_filtered)
            clean_news = []
            for val in news_list :
                clean_news.extend(val) if isinstance(val, list) else clean_news.append(val)
            clean_dict[country] = clean_news
            #reset lists
            news_list = []
            clean_news = []
        #give order to the dictionary and make it into a list
        cd = collections.OrderedDict(sorted(clean_dict.items()))
        cd_vals = list(cd.values())
        #reset new_vals
        new_vals = []
        #make items in each list into one big string
        for itemX in cd_vals :
            holderer = ' '.join(itemX)
            new_vals.append(holderer)
        return new_vals

    #returns just the sorted country names
    def country_name(self, data) :
        return sorted(data.keys())

    #puts data into correct format for classification function
    def classifier(self, text_list, big_dict):
        #split data into two lists, one for training (80%), one for testing (20%)
        train_list = text_list[:(len(text_list) // 10) * 8]
        test_list = text_list[(len(text_list) // 10) * 8:]
        #format data for training and testing
        x_train = np.array(train_list)
        y_train_text = [
            ["peace"],["peace"],["violence"],["peace"],["peace"],["peace"],["violence"],["violence"],
            ["violence"],["peace"],["violence"],["violence"],["peace"],["peace"],["violence"],["violence"],
            ["violence"],["violence"],["violence"],["violence"],["peace"],["peace"],["peace"],["peace"]
            ]
        x_test = np.array(test_list)
        target_names = ["peaceful", "violence"]
        #train classifier
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y_train_text)
        classify = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(LinearSVC()))])
        classify.fit(x_train, y)
        #test classifier
        predicted = classify.predict(x_test)
        all_labels = mlb.inverse_transform(predicted)
        #prepare country names for printing with corresponding classification
        country = self.country_name(big_dict)
        test_countries = country[(len(country) // 10) * 8:]
        #print results
        print("\nViolence status")
        for name, labels in zip(test_countries, all_labels) :
            print('{0}: {1}'.format(name, ', '.join(labels)))

#trains an algorithm to classify protests as one of 3 causes
#(government; inequality; other)
class cause_classify :

    #puts data into correct format for classification function
    def classifier(self, text_list, big_dict):
        #split data into two lists, one for training (80%), one for testing (20%)
        train_list = text_list[:(len(text_list) // 10) * 8]
        test_list = text_list[(len(text_list) // 10) * 8:]
        #format data for training and testing
        x_train = np.array(train_list)
        y_train_text = [
            ["other"],["other"],["government","inequality"],["government"],["other"],["other"],["government","inequality"],["government","inequality"],
            ["government"],["inequality","government"],["government","inequality"],["government","other"],["government"],["other"],["inequality","other"],["other","inequality"],
            ["other"],["other"],["government","inequality"],["other","government"],["government"],["other"],["other"],["other"]
            ]
        x_test = np.array(test_list)
        target_names = ["government", "inequality", "other"]
        #train classifier
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y_train_text)
        classify = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),])
        classify.fit(x_train, y)
        #test classifier
        predicted = classify.predict(x_test)
        all_labels = mlb.inverse_transform(predicted)
        #prepare country names for printing with corresponding classification
        country = violence_classify.country_name(big_dict)
        test_countries = country[(len(country) // 10) * 8:]
        #print results
        print("\nCauses")
        for name, labels in zip(test_countries, all_labels) :
            print('{0}: {1}'.format(name, ', '.join(labels)))

#get relevant output
if __name__ == '__main__' :
    locations = locations()
    data = locations.filter_data()
    locs_raw = locations.extract_locs(data)
    locs = locations.further_filter(locs_raw)
    count = locations.freq_count(locs)
    articles = locations.get_article(locs)
    violence_classify = violence_classify()
    texts = violence_classify.clean_data(articles)
    violence_classify.classifier(texts, articles)
    cause_classify = cause_classify()
    cause_classify.classifier(texts, articles)
