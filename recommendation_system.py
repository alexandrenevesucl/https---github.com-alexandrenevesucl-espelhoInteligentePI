import numpy as np
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp

def get_recommendation(emotion):
    recommendations = {
        "feliz": "Seu sorriso está ótimo! Que tal um acessório para realçar sua expressão?",
        "triste": "Parece que você está meio para baixo. Experimente ouvir sua música favorita!",
        "surpreso": "Olhos arregalados! Um delineado pode dar ainda mais destaque para sua expressão!",
        "neutro": "Seu rosto está relaxado. Talvez um toque de cor nos lábios possa dar mais vida à expressão!"
    }
    return recommendations.get(emotion, "Expressão não reconhecida, tente novamente.")

# Carregar o modelo treinado
model = load_model("emotion_model.h5")

# Inicializar a câmera
cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0) / 255.0
            
            prediction = model.predict(face)
            emotion = ["feliz", "triste", "surpreso", "neutro"][np.argmax(prediction)]
            recommendation = get_recommendation(emotion)
            
            cv2.putText(frame, recommendation, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Recomendação Baseada na Emoção", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
