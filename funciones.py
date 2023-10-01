import cv2
import numpy as np
from fpdf import FPDF
from PyPDF2 import PdfReader, PdfMerger
import os


# Ejercico 1
## Función de ecualización del histograma local
def local_hist_eq(img, window_size=[3, 3]):
    """
    Aplica ecualización local del histograma a una imagen en escala de grises.

    :param img: Imagen en escala de grises representada como un arreglo numpy.
    :param window_size: Tamaño de la ventana para la ecualización local del histograma (por defecto, [3, 3]).
    :return: Imagen con ecualización local del histograma aplicada.
    """
    # Se valida que la imagen ingresada sea un arreglo numpy con forma (x, y)
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        print(img.ndim)
        raise ValueError("La imagen debe ser un arreglo NumPy en escala de grises con forma (x, y).")

    # Se valida que el window_size ingresado sea una lista de la forma [a, b] donde a y b son números enteros
    if not isinstance(window_size, list) or len(window_size) != 2 or not all(isinstance(val, int) for val in window_size):
        raise ValueError("window_size debe ser una lista de dos números enteros [a, b].")

    # Se obtienen las dimensiones de la ventana y se comprueba que sean válidas
    window_width, window_height = window_size
    if window_width <= 1 or window_height <= 1 or window_width > img.shape[0] or window_height > img.shape[1]:
        raise ValueError("Los valores en window_size deben ser enteros mayores que 1 y menores que las dimensiones de la imagen.")

    # Se obtienen las dimensiones de la imagen ingresada
    height, width = img.shape

    # Se crea la imagen de salida inicialmente vacía del mismo tamaño que la imagen ingresada
    img_output = np.zeros((height, width), dtype=np.uint8)

    # Se calcula la mitad del ancho y alto de la ventana (vecindario) a utilizar para la ecualización local
    half_window_width = window_width // 2
    half_window_height = window_height // 2

    # Se crea un padding en la imagen original utilizando cv2.BORDER_REFLECT
    img_padded = cv2.copyMakeBorder(img, half_window_height, half_window_height, half_window_width, half_window_width, cv2.BORDER_REFLECT)

    # Bucle for para recorrer el alto de la imagen
    for y in range(height):
        # Bucle for para recorrer el ancho de la imagen
        for x in range(width):
            # 1) Se crea un recorte del tamaño de la ventana solicitada
            window = img_padded[y:y + window_height, x:x + window_width]

            # 2) Se aplica ecualización por el histograma (cv2.equalizeHist) a dicho recorte
            hist_eq_window = cv2.equalizeHist(window)

            # 3) Se obtiene el valor de intensidad del pixel central del recorte ecualizado
            central_pixel_value = hist_eq_window[half_window_height, half_window_width]

            # 4) Finalmente, se asigna ése valor en el pixel (y, x) de la imagen de salida "img_output"
            img_output[y, x] = central_pixel_value

    return img_output


# --------------------------------------------------------------------------------------------------------

# Ejercico 2
## Función para recortar formulario en celdas
def recorte_celdas_form(img):

    # Se aplica umbralización sobre la imagen
    umbral = 200
    img_umbralizada_llena = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)[1]

    # Se invierten los colores (si los bordes están en negro)
    img_umbralizada_llena_invertida = cv2.bitwise_not(img_umbralizada_llena)

    # Se detectan los contornos en la imagen umbralizada invertida
    contornos, _ = cv2.findContours(img_umbralizada_llena_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se ordenan dichos contornos por área en orden descendente
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    # Se toma el contorno de mayor área para el recorte
    mayor_contorno_llena = contornos[0]

    # Se dibuja solo el contorno de mayor área en la imagen original
    img_contorno_mayor_area = cv2.cvtColor(img_umbralizada_llena_invertida, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contorno_mayor_area, [mayor_contorno_llena], 0, (0, 0, 255), 2)

    # Se obtienen las dimensiones del mayor contorno
    x, y, w, h = cv2.boundingRect(mayor_contorno_llena)

    # Se crea una máscara en blanco del tamaño del mayor contorno
    mascara = np.zeros((h, w), dtype=np.uint8)

    # Se dibuja el contorno de mayor área en la máscara
    cv2.drawContours(mascara, [mayor_contorno_llena], -1, 255, thickness=cv2.FILLED)

    # Se recorta el interior del contorno de la imagen original
    recorte_contorno_mayor_area_llena = img_umbralizada_llena_invertida[y:y+h, x:x+w]

    # Se establecen las coordenadas en la forma x inicial, x final, y inicial, y final
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

    # Se inicializa el diccionario que contendrá las celdas
    cell_dict_lleno = {}

    # Se recorta y almacena cada celda
    for etiqueta, coordenadas in coord_celdas.items():
        x_inicial, x_final, y_inicial, y_final = coordenadas

        # Se recorta la celda de la imagen original
        celda_recortada = recorte_contorno_mayor_area_llena[y_inicial:y_final, x_inicial:x_final]

        # Se almacena la celda recortada en el diccionario
        cell_dict_lleno[etiqueta] = celda_recortada

    # Se recorre el diccionario de celdas y recortan los bordes de las imágenes
    for etiqueta, imagen in cell_dict_lleno.items():
        cell_dict_lleno[etiqueta] = imagen[5:-1, 5:-1]

    return cell_dict_lleno

## Función para determinar cantidad de caracteres y palabras
def char_n_word(imagen_grayscale):
    # Se configura el límite de área mínima y máxima
    min_area = 10  # Área mínima permitida
    max_area = 500  # Área máxima permitida

    # Se aplica connectedComponentsWithStats para encontrar componentes conexas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_grayscale)

    # Se ordenan las estadísticas por la coordenada x izquierda
    stats_ordenadas = sorted(stats, key=lambda x: x[cv2.CC_STAT_LEFT])

    # Se define el umbral de proximidad
    umbral_proximidad = 16

    # Se inicializa el contador para el número de componentes conexas dentro del límite de tamaño
    num_caracteres = 0

    # e inicializa el contador para el número de grupos de componentes
    num_palabras = 0

    # Se inicializa la lista para realizar un seguimiento de los grupos y sus componentes
    palabras = []

    # Se grafican los centroides de las componentes conexas y se cuentan las válidas
    for i, st in enumerate(stats_ordenadas):

        area = st[cv2.CC_STAT_AREA]

        # Si cumple con las restricciones de área, se grafica su centroide y se aumenta el contador
        if min_area <= area <= max_area:
            # Coordenadas del centroide
            centroide_x = st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] // 2
            num_caracteres += 1

            # Se comprueba si este caracter está cerca de la letra de una palabra existente
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

## Función de evaluación
def eval_form(cell_dict):
    # Se inicializa el diccionario de salida
    output_dict = {}

    # Se llena el diccionario de salida con el conteo de caracteres y palabras
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


    # Se corrige la ubicación de la condición Comentarios en el diccionario
    temp = output_dict['Comentarios']
    output_dict.pop('Comentarios')
    output_dict['Comentarios'] = temp
    return output_dict

## Función completa
def img_to_validation(img):

    crop_dict = recorte_celdas_form(img)

    return eval_form(crop_dict)

## Función auxiliar para armar reporte
def guardar_resultados_en_pdf(imagenes, resultados, nombre_archivo):
    # Se verifica si el archivo PDF ya existe
    if os.path.exists(nombre_archivo):
                
        # Si el PDF existe, crear uno nuevo temporal para los nuevos resultados
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        for imagen in imagenes:
            # Se guarda cada imagen
            pdf.image(f'./files/images_to_analyze/{imagen}', x=None, y=None, w=150)

            # Se agrega el nombre de la imagen como descripción
            pdf.multi_cell(0, 5, txt=f'Imagen: {imagen}', align='C')
            
            # Se arma la key de la imagen
            imagen_key = f'imagen_{imagen}'

            # Se verifica si hay resultados para esta imagen y el diccionario que le corresponde a la misma
            if imagen_key in resultados:
                diccionario = resultados[imagen_key]
                for k, v in diccionario.items():
                    pdf.multi_cell(0, 10, txt=f'{k}: {v}', align='L')
        
        # Se guarda el archivo temporal      
        pdf.output('temp.pdf')

        # Se combina el archivo original con el nuevo
        pdfMerge = PdfMerger()
        pdfiles = [nombre_archivo, './temp.pdf']
        for filename in pdfiles:
            with open(filename,'rb') as pdf:
                pdfMerge.append(PdfReader(pdf))
            os.remove(filename)

        # Se guarda el contenido completo
        pdfMerge.write(nombre_archivo)


    else:
        # Si el PDF no existe, se crea uno nuevo
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt='Listado de formularios y sus validaciones', align='C')
        for imagen in imagenes:
            # Se guarda la imagen
            pdf.image(f'./files/images_to_analyze/{imagen}', x=None, y=None, w=150)

            # Se agrega el nombre de la imagen como descripción
            pdf.multi_cell(0, 5, txt=f'Imagen: {imagen}', align='C')

            # Se arma la key del diccionario
            imagen_key = f'imagen_{imagen}'

            # Se verifica si hay resultados para esta imagen y el diccionario que le corresponde a la misma
            if imagen_key in resultados:
                diccionario = resultados[imagen_key]
                for k, v in diccionario.items():
                    pdf.multi_cell(0, 10, txt=f'{k}: {v}', align='L')
    
        # Se guarda el PDF
        pdf.output(nombre_archivo)
        


def generar_dicionario_formularios(path_imagenes, nombres_imagenes):
    resultados = {}
    for nombre_imagen in nombres_imagenes:
        image_path = path_imagenes + nombre_imagen
        # Se carga la imagen en escala de grises
        imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Se analiza la imagen con la función de validación
        resultado_dict = img_to_validation(imagen)

        # Se agrega el resultado al diccionario principal
        resultados[f'imagen_{nombre_imagen}'] = resultado_dict

        # Se mostrar el diccionario en consola
        print(f'Resultado para imagen {nombre_imagen}: {resultado_dict}')
    
    # Se devuelve el diccionario creado y los nombres de imágenes utilizados
    return resultados, nombres_imagenes


