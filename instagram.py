import os.path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import re
from nltk.stem.porter import PorterStemmer

#extract json file into python 
with open(os.path.dirname(__file__) + '../media.json') as f:
    data = json.load(f)

arr = data['photos']
arr.reverse()


# df = pd.DataFrame({'my_dates':['2015-01-01','2015-01-02','2015-01-03'],'myvals':[1,2,3]})
# df['my_dates'] = pd.to_datetime(df['my_dates'])
# df['day_of_week'] = df['my_dates'].dt.day_name()



mood = []
time = []
text = []
data = {'Text': [],
        'Label': [] }
time_ind = []
#extracting features and labels
for con in arr:
    caption = con['caption']
    if len(caption) != 0:
        if caption.find('[') != -1 and caption.find(']') != -1 and con['taken_at'] not in time:
            rank = caption[caption.index('[') + 1 : caption.index(']') ]
            description = caption[caption.index(']') + 1 : ]
            cur_time = con['taken_at']
            cur_time = cur_time[0 : 10]
            time.append(con['taken_at'])
            time_ind.append(cur_time)

            # parsing mood label into integer
            if '+' in rank:
                mood.append(int(rank))
            elif '-' in rank: 
                mood.append(int(rank))
            elif rank.isnumeric():
                mood.append(int(rank))
            else:
                continue

            # label as positive, zero, negative
            rank = int(rank)
            if rank < 0:
                rank = -1
            elif rank > 0:
                rank = 1
            else:
                rank = 0

            # # label as positive or negative; if zero assume same as most recent 
            # if int(rank) == 0:
            #     #if 0, then set as most recent non-zero
            #     for i in range(len(mood) - 1, -1, -1):
            #         if mood[i] != 0:
            #             rank = mood[i]
            #             break
            # #label set as: positive = 1, negative = 0
            # if int(rank) > 0:
            #     rank = 1
            # else:
            #     rank = 0

            text.append(description)
            data['Label'].append(int(rank))

            # text.append(description)
            # data['Label'].append(int(rank))

print(len(time))
# time_ind = list(dict.fromkeys(time_ind))
time_df = pd.to_datetime(time_ind, format = '%Y%m%d', errors='ignore')
print(time_df)

#convert to pd df
data['Text'] = text
df = pd.DataFrame(data)
texts = df['Text'].astype(str)
y = df['Label']


#using stemming to reduce word counts 
porter_stemmer = PorterStemmer()
def my_preprocessor(text):
    text = re.sub("\\W", " ", text)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return ' '.join(words)
#vectorize the text (bag of words)
#using min_df=0.01 as there are 162 documents (rows), so will be accepted if appears more than once
#strip_accents to remove Korean words
#preprocessor = my_preprocessor -> lower accuracy
vectorizer = CountVectorizer(stop_words= 'english', min_df=0.01, preprocessor=my_preprocessor)
X = vectorizer.fit_transform(texts)

#filtering out words
# print(vectorizer.get_feature_names())
print("size: " + str(X.shape))

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

model = LinearSVC(class_weight='balanced', dual= False, tol = 1e-2, max_iter= 1e5)
print(cross_val_score(model, X, y, cv=5))
print("SVC: ", cross_val_score(model, X, y, cv=5).mean())

tree = DecisionTreeClassifier(criterion = 'entropy')
print(cross_val_score(tree, X, y, cv=5))
print("tree: ", cross_val_score(tree, X, y, cv=5).mean())

guassian = GaussianNB()
print(cross_val_score(guassian, X.todense(), y, cv=5))
print("naive: ", cross_val_score(guassian, X.todense(), y, cv=5).mean())

regression = LogisticRegression()
print("regressin: ", cross_val_score(regression, X, y, cv = 5).mean())


#checking most popular words associated with good and bad mood
final_model = regression
final_model.fit(X, y)
feature_to_coef = {word: coef for word, coef in zip(vectorizer.get_feature_names(), final_model.coef_[0])}
for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
    print (best_positive)

for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
    print (best_negative)


#lemmatization of words
#http://jonathansoma.com/lede/foundations/classes/text%20processing/tf-idf/




#plotting mood against date 
#how do I reflect irregular spacing
#Use Facebook Prophet? 
# plt.plot(time, mood)
# plt.show()