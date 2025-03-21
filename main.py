import cv2

def capturar_webcam():
    cap = cv2.VideoCapture(0)  # Abre a webcam (0 é a câmera padrão)

    while True:
        ret, frame = cap.read()  # Lê um frame da webcam
        if not ret:
            break

        cv2.imshow('Espelho Inteligente', frame)  # Mostra a imagem em tempo real

        # Fecha ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capturar_webcam()
