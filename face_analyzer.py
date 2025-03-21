import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Carregar o modelo treinado
model = load_model("emotion_model.h5")

# Dicionário de emoções (mesma ordem usada no treinamento)
EMOCOES = ["feliz", "triste", "surpreso", "neutro"]

def prever_emocao(frame):
    """Redimensiona a imagem e usa o modelo treinado para prever emoções."""
    img = cv2.resize(frame, (64, 64))  # Ajusta para o tamanho do modelo
    img = np.expand_dims(img, axis=0) / 255.0  # Normaliza e adiciona dimensão

    previsao = model.predict(img)
    emocao = EMOCOES[np.argmax(previsao)]
    return emocao


def detectar_rosto(frame):
    """Detecta o rosto e retorna a análise de emoção."""
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(rgb_frame)

        if resultados.detections:
            for detection in resultados.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
                # Desenha um retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Analisa emoções com DeepFace
                emocao = prever_emocao(frame)
                cv2.putText(frame, emocao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                # Exibe a emoção detectada
                cv2.putText(frame, emocao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detectar_rosto(frame)
        cv2.imshow('Analisador Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
