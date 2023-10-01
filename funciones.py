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

def recorte_celdas_form(img):
    # Aplicar umbralización
    umbral = 200
    img_umbralizada_llena = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)[1]

    # Invertir los colores (si los bordes están en negro)
    img_umbralizada_llena_invertida = cv2.bitwise_not(img_umbralizada_llena)

    # Encontrar contornos en la imagen umbralizada invertida
    contornos, _ = cv2.findContours(img_umbralizada_llena_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos por área en orden descendente
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    # Tomar el contorno de mayor área para el recorte
    mayor_contorno_llena = contornos[0]

    # # # Dibujar solo el contorno de mayor área en la imagen original
    img_contorno_mayor_area = cv2.cvtColor(img_umbralizada_llena_invertida, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contorno_mayor_area, [mayor_contorno_llena], 0, (0, 0, 255), 2)

    # Obtener las dimensiones del mayor contorno
    x, y, w, h = cv2.boundingRect(mayor_contorno_llena)

    # Crear una máscara en blanco del tamaño del mayor contorno
    mascara = np.zeros((h, w), dtype=np.uint8)

    # Dibujar el contorno de mayor área en la máscara
    cv2.drawContours(mascara, [mayor_contorno_llena], -1, 255, thickness=cv2.FILLED)

    # Recortar el interior del contorno de la imagen original
    recorte_contorno_mayor_area_llena = img_umbralizada_llena_invertida[y:y+h, x:x+w]

    # Coordenadas en la forma x inicial, x final, y inicial, y final
    coord_celdas = {'Nombre y Apellido': (304,911,39,79),
                    'Edad': (304,911,79,119),
                    'Mail': (304,911,119,159),
                    'Legajo': (304,911,159,199),
                    'pregunta1 si': (304,608,239,279),
                    'pregunta1 no': (608,911,239,279),
                    'pregunta2 si': (304,608,279,319),
                    'pregunta2 no': (608,911,279,319),
                    'pregunta3 si': (304,608,319,359),
                    'pregunta3 no': (608,911,319,359),
                    'Comentarios' : (304,911,359,475)
                    }

    # Crear diccionario con celdas
    cell_dict_lleno = {}

    # Recortar y mostrar cada celda
    for etiqueta, coordenadas in coord_celdas.items():
        x_inicial, x_final, y_inicial, y_final = coordenadas

        # Recortar la celda de la imagen original
        celda_recortada = recorte_contorno_mayor_area_llena[y_inicial:y_final, x_inicial:x_final]

        # almacenar la celda
        cell_dict_lleno[etiqueta] = celda_recortada

    # Recorrer el diccionario y recortar los bordes de las imágenes
    for etiqueta, imagen in cell_dict_lleno.items():
        cell_dict_lleno[etiqueta] = imagen[5:-1, 5:-1]

    return cell_dict_lleno


def char_n_word(imagen_grayscale):
    # Configura el límite de área mínima y máxima según tus necesidades
    min_area = 10  # Área mínima permitida
    max_area = 500  # Área máxima permitida

    # Aplicar connectedComponentsWithStats para encontrar componentes conexas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_grayscale)

    # Ordenar las estadísticas por la coordenada x izquierda
    stats_ordenadas = sorted(stats, key=lambda x: x[cv2.CC_STAT_LEFT])

    # Umbral de proximidad
    umbral_proximidad = 16

    # Contador para el número de componentes conexas dentro del límite de tamaño
    num_caracteres = 0

    # Contador para el número de grupos de componentes
    num_palabras = 0

    # Lista para realizar un seguimiento de los grupos y sus componentes
    palabras = []

    # Graficar los centroides de las componentes conexas y contar las válidas
    for i, st in enumerate(stats_ordenadas):

        area = st[cv2.CC_STAT_AREA]

        # Si cumple con las restricciones de área, graficar su centroide y contarlo
        if min_area <= area <= max_area:
            # Coordenadas del centroide
            centroide_x = st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] // 2
            centroide_y = st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT] // 2
            num_caracteres += 1

            # Comprobar si este caracter está cerca de la letra de una palabra existente
            if len(palabras) == 0:
                palabras.append([i])
                num_palabras += 1

            elif abs(centroide_x - (stats_ordenadas[palabras[num_palabras - 1][-1]][cv2.CC_STAT_LEFT] +
                                  stats_ordenadas[palabras[num_palabras - 1][-1]][cv2.CC_STAT_WIDTH])) <= umbral_proximidad:

                palabras[num_palabras - 1].append(i)

            else:
                palabras.append([i])
                num_palabras += 1

    return {'caracteres':num_caracteres, 'palabras':num_palabras}


def eval_form(cell_dict):
    # Crear diccionario de salida
    output_dict = {}

    # Llenar el diccionario de salida con el conteo de caracteres y palabras
    for etiqueta, celda in cell_dict.items():
        output_dict[etiqueta] = char_n_word(celda)

    # Condición Nombre y Apellido
    output_dict['Nombre y Apellido'] = 'OK' if (output_dict['Nombre y Apellido']['palabras'] >= 2 and output_dict['Nombre y Apellido']['caracteres'] <= 25) else 'MAL'

    # Condición Edad
    output_dict['Edad']= 'OK' if (output_dict['Edad']['caracteres'] >= 2 and output_dict['Edad']['caracteres'] <= 3) else 'MAL'

    # Condición Mail
    output_dict['Mail'] = 'OK' if (output_dict['Mail']['palabras'] == 1 and output_dict['Mail']['caracteres'] <= 25) else 'MAL'

    # Condición Legajo
    output_dict['Legajo'] = 'OK' if (output_dict['Legajo']['caracteres'] == 8) else 'MAL'

    # Condición Comentarios
    output_dict['Comentarios'] = 'OK' if (output_dict['Comentarios']['caracteres'] <= 25) else 'MAL'

    # Condición Preguntas
    quest_nums = ['1','2','3']

    for num in quest_nums:
        dict_key = 'Pregunta ' + num
        dict_key_si = 'pregunta' + num + ' si'
        dict_key_no = 'pregunta' + num + ' no'

        if output_dict[dict_key_si]['caracteres'] > 0 and output_dict[dict_key_no]['caracteres'] > 0:
            output_dict[dict_key] = 'MAL'
        elif output_dict[dict_key_si]['caracteres'] == 1 or output_dict[dict_key_no]['caracteres'] == 1:
            output_dict[dict_key] = 'OK'
        else:
            output_dict[dict_key] = 'MAL'

        output_dict.pop(dict_key_si)
        output_dict.pop(dict_key_no)


    # Correcón de orden
    temp = output_dict['Comentarios']
    output_dict.pop('Comentarios')
    output_dict['Comentarios'] = temp
    return output_dict

def img_to_validation(img):

    crop_dict = recorte_celdas_form(img)

    return eval_form(crop_dict)


