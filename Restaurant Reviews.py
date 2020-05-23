

# Step 1.  Importing Necessary libraries
import numpy as np
import pandas as pd
import re
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Step 2.  Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

# Step 3. Simplification of Reviews by removing unnecessary words
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
for i in range(len(corpus)-599):
    print("Review no",i,"  ",corpus[i])
print('These are about 400 entries');
print("Note: Must see the corpus Variable. It is a list variable can't be seen fully on dashboard as it contains about 1000 Entries")

# Step 4.  Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X  = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Step 5. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y, test_size = 0.20,random_state = 0)

# Step 6. Import classification Model as GaussianNB
from sklearn.naive_bayes import GaussianNB
clf  = GaussianNB()
clf.fit(X_train,y_train);

# Step 7. Predicting the Test set results 
y_pred = clf.predict(X_test)

# Step 8.  Printing the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)