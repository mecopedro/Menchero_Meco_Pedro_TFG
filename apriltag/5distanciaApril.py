import cv2
import apriltag
import numpy as np
import time

# Definiendo la fuente
fonts = cv2.FONT_HERSHEY_COMPLEX

# Longitud del lado de la etiqueta AprilTag en cm
tag_size = 8.5

# Introducir parámetros intrínsecos y extrínsecos
camera_matrix = np.array([[614.3610220861610287, 0, 647.3564044852237203],
                          [0, 614.1109476093096191, 367.9357516125691063],
                          [0, 0, 1]])
dist_coeffs = np.array([0.08877443048811572979, -0.07609644217452114778,
                        0.001350161601296813885, 0.002706649165813483377,
                        0.03588047557792860276])

# Definir la longitud focal de la cámara en píxeles
focal_length = (614.3610220861610287 + 614.1109476093096191) / 2

# Crear detector de etiquetas AprilTag
detector = apriltag.Detector()

# Inicializar el objeto de la cámara para obtener frames de ella
cap = cv2.VideoCapture('test5.mkv')

# Archivo de texto para guardar las distancias
#distancia_apriltag_file = open('distancia_apriltag_test5.csv', 'w')
#distancia_apriltag_file.write('Frame, Tiempo_Frame, Distancia, Tiempo_Procesamiento\n')

# Contador de frames
frame_count = 0

# Bucle infinito para leer los frames de la cámara
while True:
    # Inicio del tiempo de procesamiento
    start_time = time.time()

    # Leer el frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Incrementar el contador de frames
    frame_count += 1

    # Calcular el tiempo del frame
    frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS)

    # Corregir la distorsión de la imagen
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar etiquetas AprilTag en el frame
    detections = detector.detect(gray)

    # Verificar si hay detecciones para procesar
    if len(detections) > 0:
        for det in detections:
            # Calcular la distancia en metros a la etiqueta detectada
            pixel_size = abs(det.corners[0][0] - det.corners[1][0])
            distancia = ((tag_size * focal_length) / pixel_size)

            # Dibujar los cuadros de delimitación alrededor de las etiquetas AprilTag detectadas
            cv2.rectangle(frame, tuple(det.corners[0].astype(int)), tuple(det.corners[2].astype(int)), (0, 255, 0), 2)
            cv2.putText(frame, f"Distancia: {distancia:.2f} cm", (int(det.corners[0][0]), int(det.corners[0][1]) - 10), fonts, 0.5, (0, 255, 0), 2)

            # Calcular el tiempo de procesamiento del frame
            processing_time = time.time() - start_time

            # Guardar datos en el archivo de texto
 #           distancia_apriltag_file.write(f"{frame_count},{frame_time:.2f},{distancia:.2f},{processing_time:.2f}\n")
    else:
        # Calcular el tiempo de procesamiento del frame
        processing_time = time.time() - start_time

        # Si no hay detecciones, registrar un error
 #       distancia_apriltag_file.write(f"{frame_count},{frame_time:.2f},No_Detection,{processing_time:.2f}\n")

    # Mostrar el frame en la pantalla
    cv2.imshow("frame", frame)

    # Comprobar si se ha presionado la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la cámara
cap.release()

# Cerrar las ventanas que están abiertas y el archivo de texto
cv2.destroyAllWindows()
#distancia_apriltag_file.close()
