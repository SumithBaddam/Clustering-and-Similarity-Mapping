from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

reviews_train = []
for line in open('full_train.txt', 'r', encoding='utf-8'):
    reviews_train.append(line.strip())
    

reviews_test = []
for line in open('full_test.txt', 'r', encoding='utf-8'):
    reviews_test.append(line.strip())


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    return reviews

reviews_train = reviews_train[:25000]
reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test = ['lc module is down', 'Power Sequencer Failure', 'cpu reset detected', 'lc module is working']
reviews_test_clean = preprocess_reviews(reviews_test)
cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print ("Final Accuracy: %s" % accuracy_score(target, final_model.predict(X_test)))

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}

pos = []
for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True):
    pos.append(best_positive)

neg = []
for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1]):
    neg.append(best_negative)

