import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# load the dataset
df = pd.read_csv('model_data.csv')

y = df.univName                 # Name of the University is our target
x = df.drop('univName',axis=1)  # Remove University name to get parameters

# Label the University names to perform SMOTE
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

x1 = pd.get_dummies(x)

# Perform over_sampling to balance the dataset
smote = SMOTE(sampling_strategy='not majority')
X1, Y1 = smote.fit_sample(x1,encoded_Y)

# Scale the values
sc = RobustScaler()       # Robust scaler takes care of outliers as well
X = sc.fit_transform(X1)

# One-hot encoding of the University Names
Y = np_utils.to_categorical(Y1)

# Make the train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

################################ Make multi-class Classifier Model #####################################################

classifier = Sequential()
classifier.add(Dense(400, activation='relu', kernel_initializer='random_normal', input_dim=X_test.shape[1]))
classifier.add(Dense(800, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(36, activation='softmax', kernel_initializer='random_normal'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train,Y_train,batch_size=20,epochs=200,verbose=0)
eval_model = classifier.evaluate(X_train,Y_train)
print("Accuracy: ",eval_model[1])     # accuracy = 0.7864

# Get the predicted class for each test sample
y_pred = classifier.predict_classes(X_test)
print(y_pred)

# Generate confusion matrix to see the performance of classifier in classifying correctly
cm = confusion_matrix(Y_test.argmax(axis=1),y_pred)
ax = plt.subplot()
sns.heatmap(cm,annot=False,ax=ax);
ax.set_xlabel('Predicted');
ax.set_ylabel('Actual');
ax.set_title('Confusion Matrix');
plt.show()

########################################## Pickle the Classifier Model #################################################

import joblib
joblib.dump(classifier, 'classifier_model.pkl')