import cv2
import mediapipe as mp
import numpy as np
import joblib

# Cargar modelo
try:
    model = joblib.load('models/rps_model.pkl')  
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    print("Asegúrate de que:")
    print("1. El archivo models/rps_model.pkl existe")
    print("2. Usaste joblib para guardar el modelo")
    exit()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

gesture_names = {0: "PIEDRA", 1: "PAPEL", 2: "TIJERAS"}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            

            landmarks_array = np.array([landmarks]).astype(np.float32)
            
            prediction = model.predict_proba(landmarks_array)[0]  
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.8:
                cv2.putText(image, f"{gesture_names[predicted_class]} ({confidence*100:.1f}%)", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Gesto no reconocido", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Piedra, Papel o Tijeras', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()