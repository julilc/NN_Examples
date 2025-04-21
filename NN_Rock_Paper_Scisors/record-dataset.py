import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import defaultdict
import shutil
from datetime import datetime

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables para almacenar datos
dataset = []
labels = []
hand_types = []  # 0=izquierda, 1=derecha
current_label = -1
gesture_names = {0: "piedra", 1: "papel", 2: "tijeras"}
hand_names = ["izquierda", "derecha"]
samples_per_combination = defaultdict(int)
target_samples = 50  # 50 por mano por gesto

# Configuración inicial
if not os.path.exists('dataset'):
    os.makedirs('dataset')

def make_backup():
    """Crea copia de seguridad del dataset existente"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for file in ['rps_dataset.npy', 'rps_labels.npy', 'rps_hand_types.npy']:
        if os.path.exists(f'dataset/{file}'):
            shutil.copy2(f'dataset/{file}', f'dataset/backup_{timestamp}_{file}')

def load_existing_data():
    """Carga datos existentes o devuelve arrays vacíos"""
    try:
        existing_data = np.load('dataset/rps_dataset.npy')
        existing_labels = np.load('dataset/rps_labels.npy')
        existing_hand_types = np.load('dataset/rps_hand_types.npy')
        return existing_data, existing_labels, existing_hand_types
    except:
        return np.array([]), np.array([]), np.array([])

def save_dataset():
    """Guarda los datos combinando con existentes (si se eligió esa opción)"""
    global dataset, labels, hand_types
    
    if menu_choice == "1":  # Nuevo dataset
        final_data = np.array(dataset)
        final_labels = np.array(labels)
        final_hand_types = np.array(hand_types)
    else:  # Agregar a existente
        existing_data, existing_labels, existing_hand_types = load_existing_data()
        final_data = np.concatenate((existing_data, np.array(dataset)))
        final_labels = np.concatenate((existing_labels, np.array(labels)))
        final_hand_types = np.concatenate((existing_hand_types, np.array(hand_types)))
    
    np.save('dataset/rps_dataset.npy', final_data)
    np.save('dataset/rps_labels.npy', final_labels)
    np.save('dataset/rps_hand_types.npy', final_hand_types)
    print(f"\nDataset guardado. Total muestras: {len(final_data)}")
    print(f"Distribución actual:\n{np.unique(final_labels, return_counts=True)}")

# Menú principal
print("\n" + "="*50)
print("SISTEMA DE CAPTURA DE GESTOS".center(50))
print("="*50)
print("\nOpciones:")
print("1. Crear NUEVO dataset (borrar existente)")
print("2. AGREGAR al dataset actual")
menu_choice = input("Seleccione (1/2): ")

if menu_choice == "1":
    make_backup()
    print("\n¡Se creará un NUEVO dataset! (Se hizo backup del anterior)")
elif menu_choice == "2":
    existing_data, _, _ = load_existing_data()
    print(f"\nAgregando al dataset existente ({len(existing_data)} muestras)")
else:
    print("\nOpción no válida. Saliendo.")
    exit()

# Captura de video
cap = cv2.VideoCapture(0)

print("\nInstrucciones:")
print("Presiona 0-2 para seleccionar gesto | s: Guardar | q: Salir")
print("Gesto actual: ninguno")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Procesamiento de imagen
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detección de manos
    current_hands = []
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_type = 0 if handedness.classification[0].label == "Left" else 1
            current_hands.append((hand_landmarks, hand_type))
    
    # Mostrar información
    info_text = [
        f"Gesto: {gesture_names.get(current_label, 'NINGUNO')}",
        f"Piedra: {samples_per_combination.get((0,0),0)}/{target_samples} (izq) | {samples_per_combination.get((0,1),0)}/{target_samples} (der)",
        f"Papel: {samples_per_combination.get((1,0),0)}/{target_samples} (izq) | {samples_per_combination.get((1,1),0)}/{target_samples} (der)",
        f"Tijeras: {samples_per_combination.get((2,0),0)}/{target_samples} (izq) | {samples_per_combination.get((2,1),0)}/{target_samples} (der)",
        "Instrucciones: 0-2: Seleccionar gesto | s: Guardar | q: Salir"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(image, text, (10, 30 + i*30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 
                   (0, 255, 0) if i == 0 else (255, 255, 255), 1)
    
    cv2.imshow('Captura de Gestos', image)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Manejo de teclas
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_dataset()
    elif key >= ord('0') and key <= ord('2'):
        current_label = key - ord('0')
        print(f"\nGesto seleccionado: {gesture_names[current_label]}")
    elif current_label != -1 and current_hands:
        for hand_landmarks, hand_type in current_hands:
            if samples_per_combination[(current_label, hand_type)] < target_samples:
                landmarks = [lm for landmark in hand_landmarks.landmark for lm in (landmark.x, landmark.y)]
                
                dataset.append(landmarks)
                labels.append(current_label)
                hand_types.append(hand_type)
                samples_per_combination[(current_label, hand_type)] += 1
                
                print(f"Capturado: {gesture_names[current_label]} - {hand_names[hand_type]} - Total: {samples_per_combination[(current_label, hand_type)]}/{target_samples}")
                time.sleep(0.3)

# Limpieza final
hands.close()
cap.release()
cv2.destroyAllWindows()