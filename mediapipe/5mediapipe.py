import cv2
import mediapipe as mp
import math
import numpy as np
import time

# Parámetros intrínsecos de la cámara
camera_matrix = np.array([[614.3610220861610287, 0, 647.3564044852237203],
                          [0, 614.1109476093096191, 367.9357516125691063],
                          [0, 0, 1]])
dist_coeffs = np.array([0.08877443048811572979, -0.07609644217452114778,
                        0.001350161601296813885, 0.002706649165813483377,
                        0.03588047557792860276])

# Definir la longitud focal de la cámara en píxeles
focal_length = (614.3610220861610287 + 614.1109476093096191) / 2

# Distancia física conocida entre dos puntos de referencia en cm (anchura de los hombros)
shoulder_width_cm = 40  

# Inicia módulo de detección de pose de Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicia la captura de video
cap = cv2.VideoCapture('test5.mkv')

# Archivo CSV para guardar las distancias
#distancia_mediapipe_file = open('distancia_mediapipe_test5.csv', 'w')
#distancia_mediapipe_file.write('Frame, Tiempo_Frame, Distancia, Tiempo_Procesamiento\n')

# Contador de frames
frame_count = 0

while True:
    # Inicio del tiempo de procesamiento
    start_time = time.time()

    # Lee el siguiente fotograma
    ret, frame = cap.read()
    if not ret:
        break

    # Incrementar el contador de frames
    frame_count += 1

    # Calcular el tiempo del frame
    frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS)

    # Corregir la distorsión de la imagen
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Obtener las coordenadas de los hombros
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        shoulder_right_x, shoulder_right_y = int(shoulder_right.x * frame.shape[1]), int(shoulder_right.y * frame.shape[0])
        shoulder_left_x, shoulder_left_y = int(shoulder_left.x * frame.shape[1]), int(shoulder_left.y * frame.shape[0])

        # Dibujar una línea entre los hombros
        cv2.line(frame, (shoulder_left_x, shoulder_left_y), (shoulder_right_x, shoulder_right_y), (0, 255, 0), 2)

        # (Opcional) Calcular la distancia a la cámara basada en la anchura de los hombros
        pixel_distance = math.sqrt((shoulder_right_x - shoulder_left_x) ** 2 + (shoulder_right_y - shoulder_left_y) ** 2)
        distance_to_camera_cm = (shoulder_width_cm * focal_length) / pixel_distance

        # Mostrar la distancia calculada
        cv2.putText(frame, f"Distance: {distance_to_camera_cm:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calcular el tiempo de procesamiento del frame
        processing_time = time.time() - start_time
        # Escribir los datos en el archivo CSV
        #distancia_mediapipe_file.write(f"{frame_count},{frame_time:.2f},{distance_to_camera_cm:.2f},{processing_time:.2f}\n")
    
    else:
        # Calcular el tiempo de procesamiento del frame
        processing_time = time.time() - start_time

        # Escribir 'No_Detection' en el archivo CSV si no se detecta una cara
        #distancia_mediapipe_file.write(f"{frame_count},{frame_time:.2f},No_Detection,{processing_time:.2f}\n")

    # Mostrar el fotograma con las detecciones y distancias    
    cv2.imshow("Frame", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

#Cerrar el archivo CSV
distancia_mediapipe_file.close()
