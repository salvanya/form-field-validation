# importar librerías necesarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from funciones import local_hist_eq

# Ruta de la imagen
image_path = 'files\images_to_analyze\Imagen_con_detalles_escondidos.tif'

# Cargar la imagen TIFF
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Imprimir la imagen
h = plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Imagen Original")
plt.colorbar(h)
plt.show(block=True)

# Tamaños de ventana
window_sizes = [[2, 2], [15, 15], [60, 60]]

# Crear una sola figura con subgráficos
fig, axes = plt.subplots(1, len(window_sizes), figsize=(15, 5))

# Aplicar la ecualización local del histograma para cada tamaño de ventana y mostrar en subgráficos
for i, window_size in enumerate(window_sizes):
    output_image = local_hist_eq(img, window_size)

    # Mostrar la imagen resultante en el subgráfico correspondiente
    axes[i].imshow(output_image, cmap='gray', vmin=0, vmax=255)
    axes[i].set_title(f"Tamaño de ventana: {window_size[0]}x{window_size[1]}")
    axes[i].axis('off')

plt.show(block=True)