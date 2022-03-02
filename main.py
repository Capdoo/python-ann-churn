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
from keras.layer import Dense






























