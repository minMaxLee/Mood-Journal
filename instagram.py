<<<<<<< HEAD
=======
import os.path
>>>>>>> second commit
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

<<<<<<< HEAD
with open('media.json') as f:
    data = json.load(f)


arr = data['photos']
arr.reverse()

#for linear regression: 
#   mood against time

#for Bag of Words
#   feature: vectorized frequency of words; label: mood
=======
#extract json file into python 
with open(os.path.dirname(__file__) + '../media.json') as f:
    data = json.load(f)

arr = data['photos']
arr.reverse()

>>>>>>> second commit
mood = []
time = []

text = []

data = {'Text': [],
        'Label': [] }

<<<<<<< HEAD
=======
#extracting features and labels
>>>>>>> second commit
for con in arr:
    caption = con['caption']
    if len(caption) != 0:
        if caption.find('[') != -1 and caption.find(']') != -1:
            rank = caption[caption.index('[') + 1 : caption.index(']')]
            description = caption[caption.index(']') + 1 : ]
            # parsing mood label into integer
            if '+' in rank:
                mood.append(int(rank))
            elif '-' in rank: 
                mood.append(int(rank))
            elif rank.isnumeric():
                mood.append(int(rank))
            else:
                continue
            cur_time = con['taken_at']
            time.append(cur_time)
            text.append(description)

<<<<<<< HEAD
            if int(rank) == 0:
                #if 0, then set as most recent non-zero
                for i in range(len(mood) - 1, -1, -1):
                    if mood[i] != 0:
                        rank = mood[i]
                        break
            #label set as: positive = 1, negative = 0
            if int(rank) > 0:
                rank = 1
            else:
                rank = 0
=======
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

>>>>>>> second commit
            data['Label'].append(int(rank))

data['Text'] = text
df = pd.DataFrame(data)
texts = df['Text'].astype(str)
y = df['Label']

#vectorize the text (bag of words)
<<<<<<< HEAD
vectorizer = CountVectorizer(stop_words='english', min_df = 0.0001)
X = vectorizer.fit_transform(texts)
#can check which words are being used! 

model = LinearSVC(class_weight='balanced', dual= False, tol = 1e-2, max_iter= 1e5)
cclf = CalibratedClassifierCV(base_estimator= model)
# cclf.fit(X,y)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.20)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(cclf, X, y, cv=5) 
=======
vectorizer = CountVectorizer(stop_words='english', min_df = 0.01)
X = vectorizer.fit_transform(texts)

#filtering out words
print(vectorizer.get_feature_names())
print("size: " + str(X.shape))
print(vectorizer.stop_words)

model = LinearSVC(class_weight='balanced', dual= False, tol = 1e-2, max_iter= 1e5)
# cclf = CalibratedClassifierCV(base_estimator= model)
# cclf.fit(X,y)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy')

from sklearn.naive_bayes import GaussianNB
guassian = GaussianNB()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# scores = cross_val_score(model, X.todense(), y, cv=5) 


>>>>>>> second commit
#need to pass in X for cclf to work, but then passing in processed data into cross_val
#manually cross val? 

# cclf.score(X_test, y_test)

#lemmatization of words
#http://jonathansoma.com/lede/foundations/classes/text%20processing/tf-idf/




#plotting mood against date 
#how do I reflect irregular spacing
#Use Facebook Prophet? 
# plt.plot(time, mood)
# plt.show()