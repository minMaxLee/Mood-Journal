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

mood = []
time = []

text = []

data = {'Text': [],
        'Label': [] }

#extracting features and labels
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
            
            time.append(con['taken_at'])
            text.append(description)

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

            data['Label'].append(int(rank))

data['Text'] = text
df = pd.DataFrame(data)
texts = df['Text'].astype(str)
y = df['Label']

porter_stemmer = PorterStemmer()
def my_preprocessor(text):
    # text = re.sub("\\W", " ", text)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return ' '.join(words)
#vectorize the text (bag of words)
#using min_df=0.01 as there are 162 documents (rows), so will be accepted if appears more than once
#strip_accents to remove Korean words
#preprocessor = 
vectorizer = CountVectorizer(preprocessor= my_preprocessor, min_df=0.01)
X = vectorizer.fit_transform(texts)

#filtering out words
print(vectorizer.get_feature_names())
print("size: " + str(X.shape))

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

model = LinearSVC(class_weight='balanced', dual= False, tol = 1e-2, max_iter= 1e5)
print(cross_val_score(model, X, y, cv=5))

tree = DecisionTreeClassifier(criterion = 'entropy')
print(cross_val_score(tree, X, y, cv=5))

guassian = GaussianNB()
print(cross_val_score(guassian, X.todense(), y, cv=5))


#lemmatization of words
#http://jonathansoma.com/lede/foundations/classes/text%20processing/tf-idf/




#plotting mood against date 
#how do I reflect irregular spacing
#Use Facebook Prophet? 
# plt.plot(time, mood)
# plt.show()