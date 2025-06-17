#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle


# In[40]:


df=pd.read_csv('diabetes.csv')
df.head()


# In[41]:


X = df.drop('Outcome', axis=1)
y = df['Outcome']
X = X.astype(float)
y = y.astype(int)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[42]:


X_train = X_train.astype(float)
y_train = y_train.astype(int)


# In[43]:


class SigmoidPerceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_output)
                # Convert probabilities to binary predictions (0 or 1)
                y_predicted = (y_predicted >= 0.5).astype(int)

                # Update rule - Use .iloc[] to access by position
                update = self.learning_rate * (y.iloc[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_output)
        return (y_predicted >= 0.5).astype(int)


# In[44]:


# Initialize and train the custom Perceptron model
sigmoid_perceptron = SigmoidPerceptron(learning_rate=0.001, max_iter=10000)
sigmoid_perceptron.fit(X_train, y_train)


# In[45]:


# Train a Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


# In[ ]:





# ### Step 6: Save both the models. Use the pickle library to save both the trained models. (naive_bayes_model.pkl and perceptron_model.pkl).

# In[46]:


def load_models():
    # Load Naive Bayes model
    nb_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))

    # Load Custom Sigmoid Perceptron model
    sigmoid_perceptron = pickle.load(open('perceptron_model.pkl', 'rb'))

    return nb_model, sigmoid_perceptron


# In[47]:


load_models()


# In[ ]:




