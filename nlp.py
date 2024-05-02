# Text PreProcessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("C:/Users/PC/Desktop/amazon_reviews.csv", sep=",")
df.head()
df.info()
df['reviewText']


########################
# Normalizing Case Folding
########################
# string ifade yer aldığı için bütün satırlar belirli bir standarta koyuldu ve büyük-küçük harf dönüşümü gerçekleşti.
df['reviewText'] = df['reviewText'].str.lower()


#########################
# Punctuations
#########################

# regular expression ( text ifadeyi noktalama işaretlerinden arındırmak için kullanılır.)
# Metinde herhangi bir noktalama işareti görüldüğünde boşluk ile değiştir.
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')


######################
# Numbers
#####################

# ilgili text içerisindeki sayıları yakala sonra boşluk ile değiştir.
df['reviewText'] = df['reviewText'].str.replace('\d', '')


######################
# Stopwords
#####################

# metinlerde herhangi bir anlamı olmayan-barınan yaygın kullanılan kelimeleri at.

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
sw = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

####################
# Rarewords
####################

# Nadir geçen kelimelerin örüntü oluşturamayacağını varsayarak onları çıkartma işlemi.
# bir kelime ne kadar sıklıkta geçiyor.

import pandas as pd
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))



#####################
# Tokenization
#####################
# cümleleri parçalamak birimleştirmek.
nltk.download('punkt')
from textblob import TextBlob
df['reviewText'].apply(lambda x: TextBlob(x).words).head()



######################
# Lemmatization
#####################

# kelimeleri köklerine indirgemek
# (stemming) ayrıca  buda bir köklerine ayırma işlemidir.
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))




###########################
# 3.Sentiment Analysis ( Duygu Durum Analizi )
###########################




# Elimizde bulunan metinlerin taşıdığı duygu durumunu matematiksel bir şekilde ifade etmeyi amaçlamaktadır.
# Sentiment Analizi için kullanılan pre-trained modellerden biri vader_lexicon.

import nltk
# SentimentIntensityAnalyzer için gerekli olan lexicon'u indirin.
nltk.download('vader_lexicon')
df['reviewText'].head()
# SentimentIntensityAnalyzer sınıfını içe aktarın.
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Şimdi SentimentIntensityAnalyzer sınıfını kullanın.
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")
sia.polarity_scores("I liked this music but it is not good as the other one")

df['reviewText'][0:10].apply(lambda x: sia.polarity_scores(x))


# polarity_score’da gelen değerlerden duygu skoru için compound dikkate alınır.
df['reviewText'][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])


# Bunu veri setinde kalıcı bir hale getirelim.
df["polarity_score"] = df['reviewText'][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df.head()



############################
# Feature Engineering
###########################



# Sınıflandırma modeli kurmak istiyoruz.

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

# Binary Target Feature oluşturuldu.
# sklearn.preprocessing modülünü yükleyin.
from sklearn import preprocessing
# LabelEncoder sınıfını içe aktarın.
from sklearn.preprocessing import LabelEncoder
# Kullanılacak makine öğrenmesi yöntemlerinin anlayacağı yönden bir binary target feature oluşuyor yani ikili hedef değişkeni.
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

# Bağımlı değişken olsun
y = df["sentiment_label"]
# Bağımsız değişken olsun
x = df["reviewText"]


#########################
# Count Vectors ( Kelimelerin frekanslarının çıkarılması)
#########################

'''
   Kelimleri matematiksel işlemlere somak için yöntemler;
   1.Count Vectors: frekans temsiller
   2.TF-IDF: normalize edilmiş frekans temsiller
   3. Word Embeddings ( Word2, Glove, BERT vs) '''

'''
   Kelimeleri nasıl temsil edeceğiz;
   - words
   kelimelerin nümerik temsilleri
   - characters
   karakterlerin nümerik temsilleri
   - ngram
   Kelime öbeklenmelerine göre özellik üretmektir.
   N-gram'lar birlikte kullanılan kelimlerin kombinasyonlarını gösterir ve feature üretmek için kullanılır
   
'''

'''      
  Count Vectors ile TF-IDF Vectors arasındaki en önemli fark frekansların standartlaştırılması.
  
'''




# Öncelikle tüm veriye tek bir metinmiş gibi yaklaşıp bütün metin içerisindeki eşsiz kelimeler çıkartılır ve bunlar sutün isimleri olur ve bunların ilgili yerlerde geçme frekansları yansıtılır.
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# Amaç her bir yorumu her bir satırı vektörleştirmek.

# word frekans
vectorizer = CountVectorizer()
x_c = vectorizer.fit_transform(corpus)

# Eşsiz kelimeler çıkartıldı.
vectorizer.get_feature_names_out()
x_c.toarray()

# n-gram frekans ( kelime öbeklerine göre )
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
x_c = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
x_c.toarray()



# Bu özellikleri şimdi veri setimize uygulayalım.
# Kelimelerin, karakterlerin ya da ngramların frekanslarını saymak için kullanılan method.
vectorizer = CountVectorizer()
x_count = vectorizer.fit_transform(x)
vectorizer.get_feature_names_out()[10:15]
x_count.toarray()[10:15]


########################
# TF-IDF ( standartlandırılmış kelime vektörü yöntemi )
########################


''' 
Kelimelerin dokümanlarda geçme frekansını ve kelimelerin bütün korpusta geçme odağında standartlaştırma işlemi yapılır.
'''
# TF ( İlgili terimin ilgili dökümandaki frekansı/dökümandaki toplam terim sayısı)
# Kelime Vektörü oluşturma yöntemi için kullanılan method TfidfVectorizer()

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizer = TfidfVectorizer()
x_tf_idf_word = tf_idf_vectorizer.fit_transform(x)


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
x_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(x)



############################
# Sentiment Modeling
###########################


# sklearn.linear_model modülünü yükleyin.
from sklearn import linear_model

# LogisticRegression sınıfını içe aktarın
from sklearn.linear_model import LogisticRegression

# Logistic Regression ( Sınıflandırma problemleri için kullanılan doğrusal formda bir sınıflandırma yöntemidir.)
log_model = LogisticRegression().fit(x_tf_idf_word, y)

# çapraz doğrulama hatası
from sklearn.model_selection import cross_val_score
cross_val_score(log_model,
                x_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()

new_review = pd.Series("this product is great")
# Öncelikle yeni gelen review dönüştürdük
# Kullanıcı tarafından yeni gelen bir yorumu modele sormadan önce uygulanan vectorize etme işlemi.
new_review = TfidfVectorizer().fit(x).transform(new_review)
log_model.predict(new_review)



# Orjinal veri seti üzerinden yorum çekip onu modele sorarsak
random_review = pd.Series(df["reviewText"].sample(1).values)
new_review = TfidfVectorizer().fit(x).transform(random_review)
log_model.predict(new_review)



#######################
# Random Forests
#######################

# Count Vectors
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(x_count, y)
cross_val_score(rf_model,
                x_count,
                y,
                cv=5,
                n_jobs=-1).mean()


# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(x_tf_idf_word, y)
cross_val_score(rf_model,
                x_tf_idf_word,
                y,
                cv=5,
                n_jobs=-1).mean()


# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(x_tf_idf_ngram, y)
cross_val_score(rf_model,
                x_tf_idf_word,
                y,
                cv=5,
                n_jobs=-1).mean()

################################
#Hiperparametre Optimizasyonu
###############################

# Önce boş bir model nesnesi oluşturalım
rf_model = RandomForestClassifier(random_state=17)

# Random forestin optimize edilmesi gereken bazı parametreleri var.
rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

from sklearn.model_selection import GridSearchCV
# Bu olası kombinasyonlarda random forestın başarısını değerlendirelim.
rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(x_count, y)

rf_best_grid.best_params_

# Bu kombinasyonlardan gelen değerler ile final modelini oluşturalım.
# Arama işlemi sonucundaki parametre değerlerini bı modele ayarlıyoruz.
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(x_count, y)

# Hatayı değerlendirelim.
cross_val_score(rf_final, x_count, y, cv=5, n_jobs=-1).mean()


















