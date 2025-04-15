import cv2
import mediapipe as mp
import numpy as np
import os
from collections import defaultdict

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables para almacenar datos
dataset = []
labels = []
current_label = -1
gesture_names = {0: "piedra", 1: "papel", 2: "tijeras"}
samples_per_gesture = defaultdict(int)
target_samples = 100  # Número de muestras por gesto

# Crear carpeta de dataset si no existe
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Función para guardar el dataset
def save_dataset():
    np.save('dataset/rps_dataset.npy', np.array(dataset))
    np.save('dataset/rps_labels.npy', np.array(labels))
    print(f"Dataset guardado con {len(dataset)} muestras")

# Captura de video
cap = cv2.VideoCapture(0)

print("Instrucciones:")
print("Presiona 0 para grabar 'piedra'")
print("Presiona 1 para grabar 'papel'")
print("Presiona 2 para grabar 'tijeras'")
print("Presiona s para guardar el dataset")
print("Presiona q para salir")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Convertir a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Dibujar landmarks
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Mostrar instrucciones
    cv2.putText(image, f"Gesto actual: {gesture_names.get(current_label, 'ninguno')}", 
                (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Muestras: Piedra={samples_per_gesture[0]}, Papel={samples_per_gesture[1]}, Tijeras={samples_per_gesture[2]}", 
                (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Instrucciones: 0-2: seleccionar gesto, s: guardar, q: salir", 
                (10, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Grabar Dataset', image)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Manejo de teclas
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_dataset()
    elif key >= ord('0') and key <= ord('2'):
        current_label = key - ord('0')
        print(f"Seleccionado: {gesture_names[current_label]}")
    elif current_label != -1 and results.multi_hand_landmarks:
        # Extraer landmarks y guardar
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        
        if samples_per_gesture[current_label] < target_samples:
            dataset.append(landmarks)
            labels.append(current_label)
            samples_per_gesture[current_label] += 1
            print(f"Guardado {gesture_names[current_label]} - Total: {samples_per_gesture[current_label]}/{target_samples}")

hands.close()
cap.release()
cv2.destroyAllWindows()