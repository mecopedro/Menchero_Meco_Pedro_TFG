# Cálculo de la distancia de un robot a una persona con visión por computador
El objetivo de este proyecto se focaliza en analizar el método más eficaz que calcule la distancia entre el robot y la persona de la que se realizará el estudio. Para ello, se estudiarán cuatro métodos de cálculo basados en las cuatro técnicas más influyentes del reconocimiento de personas que han sido desarrolladas a lo largo de la historia como son las cascadas Haar, las redes convolucionales neuronales (CNN), los histogramas de gradientes orientados (HOG) y MediaPipe.
Durante el proyecto se implementarán estos métodos utilizando la librería OpenCV de código abierto que proporciona un marco de trabajo de alto nivel para el desarrollo de aplicaciones de visión por computador en tiempo real.
## Documentación
En este repositorio se puede encontrar los códigos fuentes necesarios para la calibración de la cámara que se vaya a usar, el cálculo de la distancia mediante AprilTag que será nuestra distancia de referencia; y el cálculo de la distancia con las 4 técnicas mencionadas anteriormente.
Además, se encuentra en cada carpeta todo lo necesario para la correcta ejecución de los códigos.
Tras la ejecución de los códigos de cálculo de distancia se genera un archivo CSV con la recopilación de las distancias obtenidas por frame y tiempo de procesamiento.
Para el funcionamiento de los programas, se deberá introducir en código la ruta de los vídeos de los que se quiera calcular la distancia a la persona.
Los vídeos con los que se han probado lós códigos se encuentran en el siguiente enlace: https://upm365-my.sharepoint.com/:f:/g/personal/pedro_menchero_meco_alumnos_upm_es/EmBYZfJzVCZJsxKTLHhY-mAByTXHSLCKG5-in4mUamGNRQ?e=6avNcA
