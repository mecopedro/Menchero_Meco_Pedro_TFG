import cv2
import numpy as np

# Define el tamaño de la rejilla de calibración
# Aquí, se utiliza una rejilla de 15x10 
grid_size = (15,10)
#tamaño del cuadrado en cm
calibration_size=2.5

# Crea una matriz de puntos de objeto en el espacio 3D
# Estos puntos representan los puntos de la rejilla de calibración en el mundo real
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)

# Crea listas para almacenar los puntos de la imagen y los puntos de objeto
img_points = [] # Puntos de la imagen
obj_points = [] # Puntos del objeto

# Captura las imágenes de la cámara y detecta los puntos de la rejilla de calibración
cap = cv2.VideoCapture('calibracion.webm') # Video de calibracaion
count = 0 # Variable de conteo de imágenes capturadas
while count < 20: # Condición de salida del bucle
    # Captura un fotograma de la cámara
    ret, frame = cap.read()

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Busca los puntos de la rejilla de calibración en la imagen
    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
    
    # Dibuja los puntos de la rejilla de calibración en la imagen
    cv2.drawChessboardCorners(frame, grid_size, corners, ret)
    
    # Si se encontraron los puntos, agregalos a la lista
    key = cv2.waitKey(1) & 0xFF
    if ret == True and key == ord('c'):
        img_points.append(corners)
        obj_points.append(objp)
        
        # Incrementa la variable de conteo de imágenes capturadas
        count += 1
        
    # Muestra la imagen con los puntos dibujados
    cv2.imshow('Calibration', frame)
    
    # Imprime el número de imágenes capturadas actualmente
    print(f"Imagen {count} de 20")
    
    # Si se presiona la tecla 'q', detiene la captura de imágenes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Libera los recursos de la cámara
cap.release()
cv2.destroyAllWindows()

# Calibra la cámara utilizando los puntos de la imagen y los puntos del objeto
if count >= 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], calibration_size, None)

    # Imprime los parámetros de calibración
    print("Matriz de la cámara:")
    print(mtx)
    print("\nCoeficientes de Distorsión:")
    print(dist)
else:
    print("Se necesitan al menos 10 imágenes para la calibración de la cámara.")
    
# Guarda los parámetros de calibración en un archivo de texto
with open('calibration_params1.txt', 'w') as f:
    f.write('Matriz de la cámara:\n')
    np.savetxt(f, mtx)
    f.write('\nCoeficientes de Distorsión:\n')
    np.savetxt(f, dist)


