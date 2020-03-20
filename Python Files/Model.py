from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, GlobalAveragePooling1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import time
from Split_Data import X_train_reshaped, Y_train, X_test_reshaped, Y_test
from Metrics import f1_m, recall_m, precision_m


#define the input and output structure
n_timesteps, n_features, n_outputs = X_train_reshaped.shape[1], X_train_reshaped.shape[2], Y_train.shape[1]

	# create model
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=100, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=120, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=120, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
	# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

startTime = time.perf_counter()
# fit the model
history = model.fit(X_train_reshaped, Y_train, epochs=100, batch_size = 64, verbose=1)

endTime = time.perf_counter()
trainingTime = endTime-startTime

print("Total training time: " + str(trainingTime))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test_reshaped, Y_test, batch_size = 64, verbose=0)

print("accuracy: "+ str(accuracy))
print("LOSS: "+ str(loss))
print("precision: "+ str(precision))
print("recall: "+str(recall))
print("F1 score: "+str(f1_score))