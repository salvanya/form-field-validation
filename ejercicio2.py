import cv2
from funciones import img_to_validation

#Ruta de la imagen del formulario
image_path = 'files/images_to_analyze/formulario_04.png'

# Cargar imagen
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print(img_to_validation(img))