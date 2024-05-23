import cv2
import numpy as np
import time

# Definiendo las fuentes
fonts = cv2.FONT_HERSHEY_COMPLEX

# Definir la longitud focal de la cámara en píxeles
focal_length = (614.3610220861610287 + 614.1109476093096191) / 2

# Anchura del rostro real en cm
Known_width = 23

# Introducir parámetros intrínsecos y extrínsecos de la cámara
camera_matrix = np.array([[614.3610220861610287, 0, 647.3564044852237203],
                          [0, 614.1109476093096191, 367.9357516125691063],
                          [0, 0, 1]])
dist_coeffs = np.array([0.08877443048811572979, -0.07609644217452114778,
                        0.001350161601296813885, 0.002706649165813483377,
                        0.03588047557792860276])

# Objeto detector de caras
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Función para estimar la distancia
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

# Función para obtener datos de la cara
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_width = w
    return face_width

# Inicializar el objeto de la cámara
cap = cv2.VideoCapture('test5.mkv')

# Archivo CSV para guardar las distancias
#distancia_haarcascade_file = open('distancia_haarcascade_test5.csv', 'w')
#distancia_haarcascade_file.write('Frame, Tiempo_Frame, Distancia, Tiempo_Procesamiento\n')

# Contador de frames
frame_count = 0

# Bucle infinito para leer los frames de la cámara
while True:
    # Inicio del tiempo de procesamiento
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Incrementar el contador de frames
    frame_count += 1

    # Calcular el tiempo del frame
    frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS)

    # Corregir la distorsión de la imagen
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Obtener el ancho de la cara en el frame
    face_width_in_frame = face_data(frame)

    if face_width_in_frame != 0:
        # Calcular la distancia
        Distance = Distance_finder(focal_length, Known_width, face_width_in_frame)
       
        # Mostrar la distancia en el frame
        cv2.putText(frame, f"Distancia metodo: {round(Distance,2)}CM", (30, 35), fonts, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
        # Calcular el tiempo de procesamiento del frame
        processing_time = time.time() - start_time

        # Escribir los datos en el archivo CSV
        #distancia_haarcascade_file.write(f"{frame_count},{frame_time:.2f},{Distance:.2f},{processing_time:.2f}\n")
    else:
        # Calcular el tiempo de procesamiento del frame
        processing_time = time.time() - start_time

        # Escribir 'No_Detection' en el archivo CSV si no se detecta una cara
        #distancia_haarcascade_file.write(f"{frame_count},{frame_time:.2f},No_Detection,{processing_time:.2f}\n")

    # Mostrar el frame
    cv2.imshow("frame", frame)

    # Verificar si se presionó la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Liberar el objeto de la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

#Cerrar el archivo CSV
distancia_haarcascade_file.close()
