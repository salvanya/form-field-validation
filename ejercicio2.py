from funciones import generar_dicionario_formularios, guardar_resultados_en_pdf

## Validar los campos del formulario ##
# Ruta de la imagen del formulario
path_imagenes = './files/images_to_analyze/'

# Lista de nombres de archivo de las imágenes
nombres_imagenes =[['formulario_01.png', 'formulario_02.png', 'formulario_03.png'],['formulario_04.png', 'formulario_05.png', 'formulario_vacio.png']]

# Guardar los resultados en un archivo PDF
resultados, lista_imagenes = generar_dicionario_formularios(path_imagenes, nombres_imagenes[0])
guardar_resultados_en_pdf(lista_imagenes, resultados, './files/results/resultados_validacion_formularios.pdf')

# Actualizar el archivo PDF agregando nuevos resultados
resultados, lista_imagenes = generar_dicionario_formularios(path_imagenes, nombres_imagenes[1])
guardar_resultados_en_pdf(lista_imagenes, resultados, './files/results/resultados_validacion_formularios.pdf')

print('Resultados guardados en ./files/results/resultados_validacion_formularios.pdf')