import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import defaultdict

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables para almacenar datos
dataset = []
labels = []
hand_types = []  # Para guardar el tipo de mano
current_label = -1
gesture_names = {0: "piedra", 1: "papel", 2: "tijeras"}
hand_names = ["izquierda", "derecha"]  
samples_per_combination = defaultdict(int)  
target_samples = 100

# Crear carpeta de dataset si no existe
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Función para guardar el dataset
def save_dataset():
    np.save('dataset/rps_dataset.npy', np.array(dataset))
    np.save('dataset/rps_labels.npy', np.array(labels))
    np.save('dataset/rps_hand_types.npy', np.array(hand_types))  # Nuevo: guardar tipos de mano
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
    current_hands = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_type = 0 if handedness.classification[0].label == "Left" else 1
            current_hands.append((hand_landmarks, hand_type))
    
    # Mostrar instrucciones
    cv2.putText(image, f"Gesto actual: {gesture_names.get(current_label, 'ninguno')}", 
                (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    
    # Mostrar contadores por mano
    for i, hand_name in enumerate(hand_names):
        count = samples_per_combination[(current_label, i)] if current_label != -1 else 0
        cv2.putText(image, f"{hand_name}: {count}/{target_samples}", 
                    (10, 70 + i*30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 2)
    
    cv2.putText(image, "Instrucciones: 0-2: seleccionar gesto, s: guardar, q: salir", 
                (10, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 2)
    
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
    elif current_label != -1 and current_hands:
        for hand_landmarks, hand_type in current_hands:
            # Solo capturar si no hemos alcanzado el límite para esta combinación
            if samples_per_combination[(current_label, hand_type)] < target_samples:
                # Extraer landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                dataset.append(landmarks)
                labels.append(current_label)
                hand_types.append(hand_type)  # Guardar tipo de mano
                samples_per_combination[(current_label, hand_type)] += 1
                
                print(f"Guardado {gesture_names[current_label]} - Mano {hand_names[hand_type]} - Total: {samples_per_combination[(current_label, hand_type)]}/{target_samples}")

hands.close()
cap.release()
cv2.destroyAllWindows()