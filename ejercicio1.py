import cv2
import matplotlib.pyplot as plt
import os
from funciones import local_hist_eq

## Ecualización local del histograma ##
# Ruta de la imagen
image_path = './files/images_to_analyze/Imagen_con_detalles_escondidos.tif'

# Se carga la imagen TIFF
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Se definen los tamaños de ventana para realizar la ecualización local del histograma
window_sizes = [[2, 2], [5, 5], [10, 10], [15, 15], [20, 20],  [25, 30], [40, 40], [60, 60]]

# Se crea una figura que contendrá todas las imágenes
fig, axes = plt.subplots(3, 3, figsize=(15, 5))

# Se agrega la imagen original a la figura
row = 0
column = 0
axes[row,column].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[row,column].set_title(f"Imagen original")
axes[row,column].axis('off')

# Se aplica la ecualización local del histograma para cada tamaño de ventana y se muestran las imágenes en subgráficos
for i, window_size in enumerate(window_sizes):
    output_image = local_hist_eq(img, window_size)

    # Se definen las filas y columnas para la imagen
    column = column + 1
    if i == 2 or i == 5:
        row = row + 1
        column = 0
        
    # Se muestra la imagen resultante en el subgráfico correspondiente
    axes[row, column].imshow(output_image, cmap='gray', vmin=0, vmax=255)
    axes[row, column].set_title(f"Tamaño de ventana: {window_size[0]}x{window_size[1]}")
    axes[row, column].axis('off')

# Se guarda la figura con las distintas imágenes de salida en un archivo
file_name = 'Img_original_e_imgs_de_salida_para_distintos_tamanos_de_ventana.pdf'
file_name_png = 'Img_original_e_imgs_de_salida_para_distintos_tamanos_de_ventana.png'
plt.savefig(os.path.join('./files/results/', file_name), bbox_inches='tight')
plt.savefig(os.path.join('./files/results/', file_name_png), bbox_inches='tight')

# Se muestra la figura con las distintas imágenes de salida
plt.show()