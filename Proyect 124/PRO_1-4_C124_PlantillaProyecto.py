# Para capturar el fotograma
import cv2

# Para procesar la arreglo de la imagen
import numpy as np


# Importar el módulo tensorflow y cargar el modelo
import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')


# Adjuntando el índice de la cámara como 0 con la aplicación del software
camera = cv2.VideoCapture(0)

# Bucle infinito
while True:

	# Leyendo/Solicitando un fotograma de la cámara
	status , frame = camera.read()

	# Si somos capaces de leer exitosamente el fotograma
	if status:

		# Voltear la imagen
		frame = cv2.flip(frame , 1)
		
		
		
		# Redimensionar el fotograma
		img = cv2.resize(frame,(224,224))
		
		# Expandir las dimensiones 
		test_image = np.array(img, dtype=np.float32)
		test_image = np.expand_dims(test_image, axis=0)

		# Normalizar antes de alimentar al modelo
		normalised_image = test_image/255.0

		# Obtener predicciones del modelo
		prediction = model.predict(normalised_image)
		
		
		# Mostrando los fotogramas capturados
		cv2.imshow('Alimentar' , frame)

		# Esperando 1ms
		code = cv2.waitKey(1)
		
		# Si se preciona la barra espaciadora, romper el bucle
		if code == 32:
			break

# Liberar la cámara de la aplicación del software
camera.release()

# Cerrar la ventana abierta
cv2.destroyAllWindows()
