import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Carga de dataset
try:
    X = np.load('dataset/rps_dataset.npy')
    y = np.load('dataset/rps_labels.npy')
except:
    print("Error: No se encontraron los archivos del dataset. Ejecuta primero record-dataset.py")
    exit()

# Preprocesamiento
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=50,
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train)

# Evaluar
accuracy = model.score(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy*100:.2f}%")

# Guardar el modelo
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(model, 'models/rps_model.pkl')
print("Modelo guardado como models/rps_model.pkl")