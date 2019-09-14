###########Sentiment analysis LSTM for unique patterns##############
import pandas as pd
from keras.preprocessing.text import Tokenizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import keras 
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words('english'))
#Check if error message has any keyword, fail, failure, err, error, invalid
#unique_patterns = pd.read_csv('Unique_Patterns_ASR9000_2017.csv', encoding='latin-1')
unique_patterns = pd.read_csv('trainingSet_SR.csv', encoding='iso-8859-1')
#unique_patterns = unique_patterns[unique_patterns['Ignore']==0]
unique_patterns["codeStr"].fillna("", inplace=True)

signature_string = unique_patterns['codeStr'].tolist()
y = unique_patterns['label_predictions'].tolist()

docs_processed=[]
for doc in signature_string:
    doc = re.sub(' +',' ',doc)
    regex = r"\S* # \S*\s?"
    doc = re.sub(regex, '', doc, 0)
    regex = r"\S*# \S*\s?"
    doc = re.sub(regex, '', doc, 0)
    #regex = r'\w*[0-9]\w*'
    #doc = re.sub(regex, '', doc, 0)
    #remove dates and time
    regex = r"\d\d-\S*-\d\d\d\d*"
    doc = re.sub(regex, '', doc, 0)
    doc = re.sub(' +',' ',doc)
    #Remove outlook added lines
    regex = r"_"
    doc = re.sub(regex,' ',doc)
    regex = r"\-\-\s*\n"
    doc = re.sub(regex,'',doc)
    doc = doc.replace('*', ' ')
    doc = doc.replace('=', ' ')
    doc = doc.replace('__', ' ')
    doc = doc.replace('..', ' ')
    doc = doc.replace(';', ' ')
    doc = doc.replace('"', ' ')
    doc = doc.replace(r'[^\x00-\x7F]+', ' ')
    doc = doc.replace('(', ' ')
    doc = doc.replace(')', ' ')
    doc = doc.replace('[', ' ')
    doc = doc.replace(']', ' ')
    doc = doc.replace('+', ' ')
    doc = doc.replace('|', ' ')
    doc = doc.replace('#', ' ')
    doc = re.sub(r"==", " ", doc, 0)
    doc = re.sub(r"-", " ", doc, 0)
    doc = re.sub(r"&", " ", doc, 0)
    #doc = re.sub(r".", " ", doc, 0)
    doc = re.sub(r"--", " ", doc, 0)
    doc = re.sub(r"{", " ", doc, 0)
    doc = re.sub(r"}", " ", doc, 0)
    doc = re.sub(r":", " ", doc, 0)
    doc = re.sub(r"/", " ", doc, 0)
    doc = re.sub(r">", " ", doc, 0)
    doc = re.sub(r"<", " ", doc, 0)
    doc = re.sub(r",", " ", doc, 0)
    doc = re.sub(r"'", " ", doc, 0)
    doc = re.sub(r"!", " ", doc, 0)
    doc = re.sub(r"@", " ", doc, 0)
    doc = re.sub(r"GMT", " ", doc, 0)
    docs_processed.append(' '.join(doc.split()))

lemma = WordNetLemmatizer()

def get_pos( word ):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


def clean_text(inputStr):
    #a = inputStr.split(' ')
    y = []
    #print(inputStr)
    for word in inputStr:
        status = True
        for w in word.split(' '):
            #if(lemma.lemmatize(w, get_pos(w)).lower() in stop): #or lemma.lemmatize(w, get_pos(w)).lower() in stoplist):
                #print(w, 1)
            #if(lemma.stem(w).lower() in stop or lemma.stem(w).lower() in stoplist):
                #status = False
            #if(w.lower() in stop): #or w.lower() in stoplist):
                #print(w,2)
                #status = False
            if(w.isalpha() == False):
                #print(w, 3)
                #print(w)
                status = False
            if(len(w) > 0):
                if(ord(w[0]) > 120):
                    status = False
                    #print(w)
        if(status == True and len(word) > 2):
            #print(word)
            #print(word.isalpha())
            #y.append(lemma.lemmatize(word, get_pos(word)).lower())
            y.append(word.lower().strip())
    #return list(set(y))
    return ' '.join(y)

docs_clean = [clean_text(doc.split(' ')) for doc in docs_processed]
x = docs_clean


cv = CountVectorizer(binary=True)
cv.fit(docs_clean)

pickle.dump(cv.vocabulary_,open("count_vectorizer.pkl","wb"))

X = cv.transform(docs_clean)
X_test = cv.transform(docs_clean[2000:])
X_pred = cv.transform(docs_clean)
target = y
y_test = y[2000:]
#target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=0.5)
final_model.fit(X_train, y_train)

predictions1 = final_model.predict(X_test)
print(predictions1[10:20])
print(y_test[10:20])

#Saving the model
pkl_filename = "sentiment_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(final_model, file)


#unique_patterns['Predictions'] = predictions
#unique_patterns.to_csv('sentiment_analysis_predictions.csv', encoding='utf-8')

#preds = pd.DataFrame()
#preds['prediction'] = predictions
#preds['signature_strings'] = signature_string[10000:]

'''
tokenizer = Tokenizer(num_words=2500,split=' ')
tokenizer.fit_on_texts(x)


X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X)


Y = []
for val in y:
    if(val == 0):
        Y.append([1,0])
    else:
        Y.append([0,1])

Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.8)

model = Sequential()
model.add(Embedding(2500,128,input_length=X.shape[1],dropout=0.2))
model.add(LSTM(300, dropout_U=0.2,dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=6,verbose=2,batch_size=32)
print(model.evaluate(x_test,y_test)[1])

preds2 = model.predict(X)
predictions2 = []
for i in preds2:
	if(i[0] > i[1]):
		predictions2.append(0)
	else:
		predictions2.append(1)
'''
unique_patterns['Predictions_lr'] = predictions1
#unique_patterns['Predictions_lstm'] = predictions2
unique_patterns.to_csv('sentiment_analysis_predictions.csv', encoding='utf-8')


############TESTING#############
#Automate to run on all PFs...

unique_patterns = pd.read_csv('SRSignatures_ASR9000.csv', encoding='iso-8859-1')
unique_patterns["codeStr"].fillna("", inplace=True)

signature_string = unique_patterns['codeStr'].tolist()
#y = unique_patterns['label_predictions'].tolist()

docs_processed=[]
for doc in signature_string:
    doc = re.sub(' +',' ',doc)
    regex = r"\S* # \S*\s?"
    doc = re.sub(regex, '', doc, 0)
    regex = r"\S*# \S*\s?"
    doc = re.sub(regex, '', doc, 0)
    #regex = r'\w*[0-9]\w*'
    #doc = re.sub(regex, '', doc, 0)
    #remove dates and time
    regex = r"\d\d-\S*-\d\d\d\d*"
    doc = re.sub(regex, '', doc, 0)
    doc = re.sub(' +',' ',doc)
    #Remove outlook added lines
    regex = r"_"
    doc = re.sub(regex,' ',doc)
    regex = r"\-\-\s*\n"
    doc = re.sub(regex,'',doc)
    doc = doc.replace('*', ' ')
    doc = doc.replace('=', ' ')
    doc = doc.replace('__', ' ')
    doc = doc.replace('..', ' ')
    doc = doc.replace(';', ' ')
    doc = doc.replace('"', ' ')
    doc = doc.replace(r'[^\x00-\x7F]+', ' ')
    doc = doc.replace('(', ' ')
    doc = doc.replace(')', ' ')
    doc = doc.replace('[', ' ')
    doc = doc.replace(']', ' ')
    doc = doc.replace('+', ' ')
    doc = doc.replace('|', ' ')
    doc = doc.replace('#', ' ')
    doc = re.sub(r"==", " ", doc, 0)
    doc = re.sub(r"-", " ", doc, 0)
    doc = re.sub(r"&", " ", doc, 0)
    #doc = re.sub(r".", " ", doc, 0)
    doc = re.sub(r"--", " ", doc, 0)
    doc = re.sub(r"{", " ", doc, 0)
    doc = re.sub(r"}", " ", doc, 0)
    doc = re.sub(r":", " ", doc, 0)
    doc = re.sub(r"/", " ", doc, 0)
    doc = re.sub(r">", " ", doc, 0)
    doc = re.sub(r"<", " ", doc, 0)
    doc = re.sub(r",", " ", doc, 0)
    doc = re.sub(r"'", " ", doc, 0)
    doc = re.sub(r"!", " ", doc, 0)
    doc = re.sub(r"@", " ", doc, 0)
    doc = re.sub(r"GMT", " ", doc, 0)
    docs_processed.append(' '.join(doc.split()))

lemma = WordNetLemmatizer()

def get_pos( word ):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


def clean_text(inputStr):
    #a = inputStr.split(' ')
    y = []
    #print(inputStr)
    for word in inputStr:
        status = True
        for w in word.split(' '):
            #if(lemma.lemmatize(w, get_pos(w)).lower() in stop): #or lemma.lemmatize(w, get_pos(w)).lower() in stoplist):
                #print(w, 1)
            #if(lemma.stem(w).lower() in stop or lemma.stem(w).lower() in stoplist):
                #status = False
            #if(w.lower() in stop): #or w.lower() in stoplist):
                #print(w,2)
                #status = False
            if(w.isalpha() == False):
                #print(w, 3)
                #print(w)
                status = False
            if(len(w) > 0):
                if(ord(w[0]) > 120):
                    status = False
                    #print(w)
        if(status == True and len(word) > 2):
            #print(word)
            #print(word.isalpha())
            #y.append(lemma.lemmatize(word, get_pos(word)).lower())
            y.append(word.lower().strip())
    #return list(set(y))
    return ' '.join(y)

docs_clean = [clean_text(doc.split(' ')) for doc in docs_processed]
x = docs_clean

###Model 1 predictions###
X_pred = cv.transform(docs_clean)

# Load from file
pkl_filename = "sentiment_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    final_model = pickle.load(file)

predictions1 = final_model.predict(X_pred)
'''
###Model 2 predictions###
X = tokenizer.texts_to_sequences(x)
for i in X:
	if(len(i) > maxlen):
		maxlen = len(i)

X = pad_sequences(X)

preds2 = model.predict(X)
predictions2 = []
for i in preds2:
	if(i[0] > i[1]):
		predictions2.append(0)
	else:
		predictions2.append(1)
'''
unique_patterns['Predictions'] = predictions1
#unique_patterns['Predictions_lstm'] = predictions2
unique_patterns.to_csv('SRSignatures_ASR9000_labeled.csv', encoding='utf-8', index =False)
unique_patterns.to_csv('SRSignatures_ASR9000_labeled.csv', encoding='utf-8', index =False)
