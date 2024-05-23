import cv2
import numpy as np
import time

# Parámetros de calibración de la cámara
camera_matrix = np.array([[614.3610220861610287, 0, 647.3564044852237203],
                          [0, 614.1109476093096191, 367.9357516125691063],
                          [0, 0, 1]])
dist_coeffs = np.array([0.08877443048811572979, -0.07609644217452114778,
                        0.001350161601296813885, 0.002706649165813483377,
                        0.03588047557792860276])

# Calcula la longitud focal promedio
focal_length = (614.3610220861610287 + 614.1109476093096191) / 2

# Altura estimada de una persona en centimetros
real_height = 175  

# Clases que el modelo puede reconocer
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Cargar el modelo preentrenado de MobileNetSSD
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Función para calcular la distancia a un objeto basándose en la altura en píxeles y la altura real
def calculate_distance(pixel_height, real_height, focal_length):
    return (real_height * focal_length) / pixel_height

# Inicialización del video y archivo CSV para guardar los resultados
cap = cv2.VideoCapture('test7.mkv')
#distancia_cnn_file = open('distancia_cnn_test7.csv', 'w')
#distancia_cnn_file.write('Frame, Tiempo_Frame, Distancia, Tiempo_Procesamiento\n')

# Inicialización de variables de control del procesamiento
frame_count = 0

# Bucle principal de procesamiento de video
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS)
    # Corregir la distorsión de la lente en el cuadro
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Preparar el cuadro para la red neuronal
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Recorrer las detecciones
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filtrar detecciones por confianza y clase
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            # Calcular y mostrar la distancia
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            pixel_height = endY - startY
            distance = calculate_distance(pixel_height, real_height, focal_length)
            
            processing_time = time.time() - start_time
 #           distancia_cnn_file.write(f"{frame_count},{frame_time:.2f},{distance:.2f},{processing_time:.2f}\n")
            
            # Mostrar la distancia estimada y la caja alrededor de la persona detectada
            cv2.putText(frame, f"Distancia: {distance:.2f}m", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            break  # Solo procesar la primera detección de persona

    # Mostrar el cuadro y terminar el proceso si se presiona 'q'
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
#distancia_cnn_file.close()
