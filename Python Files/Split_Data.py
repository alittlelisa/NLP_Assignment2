from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import newaxis
from Process_data import documents

all_data = pd.DataFrame(documents, columns=['Phrase','Sentiment'])

#converts the labels to categorical/onehot encoded for multiclass predictions
Y_categorical = np_utils.to_categorical(all_data["Sentiment"])

#stratify the data to keep the categories evenly distributed for better results
X_train, X_test, Y_train, Y_test = train_test_split(all_data["Phrase"],  all_data["Sentiment"], test_size=0.3, stratify = Y_categorical, random_state=2003)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range = (1, 1))

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

#reshapes the training and testing inputs as proper input to the model
X_train_np = X_train.toarray()
X_train_reshaped = X_train_np[..., newaxis]
X_test_np = X_test.toarray()
X_test_reshaped = X_test_np[..., newaxis]


