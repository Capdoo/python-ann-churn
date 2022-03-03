#Librerias
import pandas as pd
import numpy as np

# 1. Dataset
dataset = pd.read_csv("Churn_Modelling.csv")


# 2. Preprocesamiento
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values


## Tratamiento variables Dummy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

## Paises
columnTransformer = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder='passthrough')
X = columnTransformer.fit_transform(X)
## Eliminando columna de más
X = X[:, 1:]

### Genero
labelenconder_X = LabelEncoder()
X[:,3] = labelenconder_X.fit_transform(X[:,3]) 


# 3. Division dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## 3.1 Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# 4. Construcción de la Red Neuronal
import keras
from keras.models import Sequential
from keras.layers import Dense

##Secuential para iniciar la red neuronal
classifier = Sequential()

##Añadir la capa de entrada y oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim = 11))

## Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu"))

##Añadir capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))


## Compilar la red neuronal
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


### Ajustar la ANN con el entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# 5. Evaluación de la ANN
y_pred = classifier.predict(X_test)

## Usando un umbral
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print((1548+134)/2000)





















