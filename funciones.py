import cv2
import numpy as np
import matplotlib.pyplot as plt

def local_hist_eq(img, window_size=[3, 3]):
    """
    Aplica ecualización local del histograma a una imagen en escala de grises.

    :param img: Imagen en escala de grises representada como un arreglo numpy.
    :param window_size: Tamaño de la ventana para la ecualización local del histograma (por defecto, [3, 3]).
    :return: Imagen con ecualización local del histograma aplicada.
    """
    # Validar que img sea un arreglo numpy con forma (x, y)
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError("La imagen debe ser un arreglo NumPy en escala de grises con forma (x, y).")

    # Validar que window_size sea una lista de la forma [a, b] donde a y b son números enteros
    if not isinstance(window_size, list) or len(window_size) != 2 or not all(isinstance(val, int) for val in window_size):
        raise ValueError("window_size debe ser una lista de dos números enteros [a, b].")

    a, b = window_size
    if not isinstance(a, int) or not isinstance(b, int) or a <= 1 or b <= 1 or a > img.shape[0] or b > img.shape[1]:
        raise ValueError("Los valores en window_size deben ser enteros mayores que 1 y menores que las dimensiones de la imagen.")

    # Obtener las dimensiones de la imagen
    height, width = img.shape

    # Crear una imagen de salida inicialmente vacía del mismo tamaño que "img"
    img_output = np.zeros((height, width), dtype=np.uint8)

    # Obtener las dimensiones de la ventana
    window_width, window_height = window_size

    # Calcular la mitad del ancho y alto de las ventanas (ventana cuadrada)
    half_window_width = window_width // 2
    half_window_height = window_width // 2

    # Crear padding en la imagen original utilizando cv2.BORDER_REFLECT
    img_padded = cv2.copyMakeBorder(img, half_window_height, half_window_height, half_window_width, half_window_width, cv2.BORDER_REFLECT)

    # Bucle for para recorrer el alto de la imagen
    for y in range(height):
        # Bucle for para recorrer el ancho de la imagen
        for x in range(width):
            # 1) Crear un recorte del tamaño de la ventana
            window = img_padded[y:y + window_height, x:x + window_width]

            # 2) Aplicar ecualización por el histograma (cv2.equalizeHist) al recorte realizado
            hist_eq_window = cv2.equalizeHist(window)

            # 3) Obtener el valor de intensidad del pixel central del recorte ecualizado
            central_pixel_value = hist_eq_window[half_window_height, half_window_width]

            # 4) Asignar ese valor en el pixel (y, x) de la imagen de salida "img_output"
            img_output[y, x] = central_pixel_value

    return img_output


