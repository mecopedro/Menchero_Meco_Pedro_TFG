import cv2
import numpy as np
import time

# Introducimos parámetros intrínsecos
camera_matrix = np.array([[613.7311977825676195, 0, 643.3180619040773536], [0, 613.7075678610633531, 367.9948408035565990], [0, 0, 1]])
dist_coeffs = np.array([0.08436089228533759365, -0.06024711709487337585, 0.001346608486215684474, 0.0007103137372465075094, 0.03427745281187183357])

# Definir la longitud focal de la cámara en píxeles
focal_length = (614.3610220861610287 + 614.1109476093096191) / 2

# Altura real de la persona en cm
person_height = 175

# Configuración de detección de persona HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Cargar el video
cap = cv2.VideoCapture('test7.mkv')

# Archivo CSV para guardar las distancias
#distancia_hog_file = open('distancia_hog_test7.csv', 'w')
#distancia_hog_file.write('Frame, Tiempo_Frame, Distancia, Tiempo_Procesamiento\n')

# Contador de frames
frame_count = 0

# Loop principal
while True:
    # Inicio del tiempo de procesamiento
    start_time = time.time()
    # Leer el frame del video
    ret, frame = cap.read()
    
    # Verificar si el frame fue leído correctamente
    if not ret:
        break

    # Incrementar el contador de frames
    frame_count += 1

    # Calcular el tiempo del frame
    frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS)
    
    # Corregir la distorsión de la imagen
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Detección de persona HOG
    rects, _ = hog.detectMultiScale(frame)
    
    # Verificar si hay detecciones
    if len(rects) > 0:
        # Tomar la detección más grande (asumiendo que es la más cercana)
        largest_rect = max(rects, key=lambda r: r[2] * r[3])
        rects = [largest_rect]
        
        # Dibujar el rectángulo y calcular distancia
        for rect in rects:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

            # Calcular la distancia a la cámara
            distance = (focal_length * person_height) / rect[3]
            cv2.putText(frame, f"{distance:.2f} m", (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Escribir los datos en el archivo CSV
            processing_time = time.time() - start_time
            #distancia_hog_file.write(f"{frame_count},{frame_time:.2f},{distance:.2f},{processing_time:.2f}\n")
    else:
        # No hay detecciones
        processing_time = time.time() - start_time
        #distancia_hog_file.write(f"{frame_count},{frame_time:.2f},No_Detection,{processing_time:.2f}\n")
    
    # Mostrar el frame
    cv2.imshow("frame", frame)

    # Verificar si se presionó la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

# Cerrar el archivo CSV
distancia_hog_file.close()
