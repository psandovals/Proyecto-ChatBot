#/*/*/*/*/LIBRERIAS UTILIZADAS/*/*/*/*/*/*/*/
#
import nltk as nt
#LIBRERÍA PARA USO Y MANEJO DE REDES NEURONALES
import tensorflow as tf
#LIBRERÍA INTERFAZ Y MANIPULACION DE TENSORFLOW
import tflearn as tl
#LIBRERÍA PARA MANEJO DE ARREGLOS Y MATRICES
import numpy as np 
#LIBRERÍA PARA MANIPULACIÓN DE ARCHIVOS JSON
import json 
#LIBRERÍA PARA GENERACIÓN DE DATOS RANDOM
import random 
#LIBRERÍA PARA GUARDAR NUESTROS MODELOS DE REDES NEURONALES
import pickle

#/*/*/*/*/*/*/*/INSTANCIA CLASE "LancasterStemmer"/*/*/*/*/*/*/*/
 #CREAMOS UN OBJETO DE LA CLASE lancasterStemmer
 #CON SU METODO stem NOS PERMITE OBTENER LA RAÍZ LEXICA DE UNA PALABRA
stemmer = nt.stem.lancaster.LancasterStemmer()    

#/*/*/*/*/*/*/*/LECTURA DE ARCHIVO JSON/*/*/*/*/*/*/*/
#OBTENEMOS LOS DATOS DE NUESTRO ARCHIVO JSON
with open("respuestas.json") as f:
    #VOLCAMOS LOS DATOS DEL ARCHIVO JSON EN UNA VARIABLE
    archivoJson = json.load(f)
#/*/*/*/*/*/*/ DECLARACIÓN LISTAS/*/*/*/*/*/*
#LISTA QUE CONTIENE LAS PALABRAS RECONOCIDAS
palabras = []
#LISTA QUE CONTIENE LAS ETIQUETAS DE LAS PALABRAS
tags = []
#LISTAS AUXILIARES
auxX=[]
auxY=[]

#/*/*/*/*/*/*/RECORRIDO ARCHIVO JSON Y LLENADO DE VARIABLES(patrones y tags)/*/*/*/*/*/*
#RECORREMOS EL ARCHIVO JSON EN EL ARREGLO DE CATEGORÍAS
for i in archivoJson["respuestas"]:
    #RECORREMOS EL ARCHIVO JSON EN EL ARREGLO DE PATRONES
    for patrones in i["patrones"]:
        #OBTENEMOS LAS FRASES DEL ARREGLO DE PATRONES Y RECONOCE LAS PALABRAS
        #RECONOCIMIENTO DE PALABRAS CON word_tokenize 
        auxPalabra = nt.word_tokenize(patrones)
        #INSERTAMOS LAS PALABRAS RECONOCIDAS EN TOKENS EN palabras
        palabras.extend(auxPalabra)
        #INSERTAMOS LAS PALABRAS RECONOCIDAS EN TOKENS EN auxX
        auxX.append(auxPalabra)
        #INSERTAMOS LOS TAG DE NUESTRO JSON EN auxY
        auxY.append(i["tag"])

        #CONDICIONAL SI EL TAG NO ESTA EN NUESTRO ARREGLO tags
        if i["tag"] not in tags:
            #INSERTAMOS ESE TAG EN EL ARREGLO tags
            tags.append(i["tag"])


#/*/*/*/*/*/*/ PROCESO STEM EN PATRONES TOKENIZADOS/*/*/*/*/*/*
#EL ARREGLO PALABRAS CONTIENE LOS TOKENS Y BUSCAMOS LAS RAÍCES DE LAS PALABRAS 
#Y EVITAMOS LOS SIGNOS DE INTERROGACIÓN
palabras = [stemmer.stem(p.lower()) for p in palabras if p != "?"] 
#DEVOLVEMOS UNA LISTA ORDENADA DE palabras
palabras = sorted(list(set(palabras)))
#DEVOLVEMOS UNA LISTA ORDENADA DE tags
tags = sorted(tags)

#/*/*/*/*/*/*/DECLARACIÓN LISTAS/*/*/*/*/*/*
#CREAMOS LISTA PARA ENETRENAMIENTO DE LA RED NEURONAL
entrenamiento = []
#CREAMOS LISTA PARA LOS DATOS DE SALIDA
salida = []
#CREAMOS UNA LISTA LLENA DE LA CANTIDAD DE CEROS COMO LA LONGITUD DE LA LISTA tags
salidaVacia = [0 for _ in range(len(tags))]



#/*/*/*/*/*/*/*/APLICACIÓN DEL ALGORITMO DE LA CUBETA/*/*/*/*/*/*//*/
for x,documento in enumerate(auxX):
    #CREAMOS UNA LISTA VACIA LLAMADA cubeta
    cubeta= []
    #OBTENEMOS LA RAÍZ DE LAS PALABRAS DE LA VARIABLE documento
    auxPalabra = [stemmer.stem(p.lower()) for p in documento]

    #RECORREMOS LA LISTA palabras
    for w in palabras : 
        #SI EL CONTENIDO COINCIDE CON EL COTENIDO DE auxpalabra
        #EN LA LISTA CUBETA INSERTAMOS UN 1
        if w in auxPalabra:
            cubeta.append(1)
        #DE LO CONTRARO INSERTAMOS EN LA LISTA CUBETA UN 0
        else:
            cubeta.append(0)
    #ASIGMANOS A filaSalida EL CONTENIDO DE salidaVacia
    filaSalida = salidaVacia[:]
    #OBTENEMOS EL ELEMENTO DE auxY EN LA POSICION x, 
    #LUEGO OBTENEMOS EL INDICE DE ESE ELEMENTO EN LA LISTA tags 
    #PARA LUEGO ASIGNARLE UN 1
    filaSalida[tags.index(auxY[x])] = 1
    #INSERTAMOS EL RESULTADO DE LA LISTA cubeta en lA LISTA entrenamiento
    entrenamiento.append(cubeta)
    #INSERTAMOS EL RESULTADO DE LA LISTA filaSalida en salida
    salida.append(filaSalida)

#/*/*/*/**/*/CONVERTIMOS LAS 2 LISTAS EN ARRAYS DE NUMPY/*/*/*/*/*/*/*/*/*/*
entrenamiento = np.array(entrenamiento)
salida = np.array(salida)

#/*/*/*/*/*/*/DECLARACION DE NUESTRAS CAPAS DE RED NEURONAL/*/*/*/*/*/*
#DEFINIMOS LAS ENTRADAS DE NUESTRA RED NEURONAL SIN NINGUNA FORMA Y CON LA LONGITUD EN 0 DE LA MATRIZ "entrenamiento"
net = tl.input_data(shape = [None, len(entrenamiento[0]) ])
#DEFINIMOS LA PRIMERA HIDDEN LAYER DE NUESTRA RED TOTALMENTE CONECTADA CON TODOS LOS NODOS Y CON UNA CANTIDAD DE 20
net = tl.fully_connected(net, 100)
#DEFINIMOS LA SEGUNDA HIDDEN LAYER DE NUESTRA RED TOTALMENTE CONECTADA CON TODOS LOS NODOS Y CON UNA CANTIDAD DE 20
net = tl.fully_connected(net, 100)
##DEFINIMOS LA TERCERA HIDDEN LAYER DE NUESTRA RED TOTALMENTE CONECTADA CON TODOS LOS NODOS Y CON UNA CANTIDAD DE 20
net = tl.fully_connected(net, 100)
#DEFINIMOS LAS SALIDAS DE NUESTRA RED NEUROAL CON LA LONGITUD DEL ARRAY "salida" Y CON EL TIPO DE ACTIVACIÓN SOFTMAX 
net = tl.fully_connected(net, len(salida[0]), activation = "softmax")
#APLICAMOS REGRESION PARA OBTENER PROBABILIDADES DE NUESTRA RED NEURONAL
net = tl.regression(net)
#ESTABLECEMOS EL MODELO DE NUESTRA RED NEURONAL Y ENVIAMOS NUESTRA RED COMO PARÁMETRO

#/*/*/*/*/*/*/DECALARACION Y CONFIGURACION DE MODELO DE RED NEURONAL/*/*/*/*/*/*
model = tl.DNN(net)
try:
    model.load("modelo.tflearn")
except:
    #DAMOS FORMA DE A NUESTRA RED NEURONAL CON PARAMETTROS DE: ENTRADA(etrenamiento), OBJETIVSO(salida), 
    # EPOCH(numero de repeticiones de entrenamiento),BATCH_SIZE(numero de muestras/lote propagada en la red)
    #, SHOW_METRIC(aceptamos mostrar las metricas de las iteraciones)
    model.fit(entrenamiento, salida, n_epoch = 50000, batch_size = len(palabras[0]), show_metric= True)
    #GUARDAMOS NUESTRO MODELO ENTRENADO
    model.save("modelo.tflearn")

#/*/*/*/**/*/*/*/*/*/INTERACCION USUSARIO/*/*/*/*/*/*/*/*/*/
#DEGINIMOS NUESTRA FUNCION PRINCIPAL DE INTERACCION DE CHAT CON EL USUARIO
def mainChatBot():
    #CREAMOS UN CICLO INFINITO
    while True:
        #/*/*/*/*/*/TRATAMIENTO DE ENTRADA DEL USUARIO/*/*/*/*/
        #ASIGNAMOS LA ENTRADA DE DATOS A "entrada" Y MOSTRAMOS EN PANTALLA "Tu" PARA EL USUARIO
        entrada = input("Tu:")
        #LLENAMOS DE 0  LA CUBETA CON LA LONGITUD DE LA LISTA "palabras"
        cubeta = [0 for _ in range(len(palabras))]
        #SEPARAMAOS EN TOKENS EL STRING QUE RECIBE "entrada"
        entradaProc = nt.word_tokenize(entrada)
        #OBTENEMOS LA RAIZ DE LAS PALABRAS SEPARADAS EN TOKENS CON EL METODO "stem"
        entradaProc = [stemmer.stem(p.lower()) for p in entradaProc]
        #RECORREMOS EL ARREGLO "entradaProc"
        for palabraIndividual in entradaProc:
            #REALIZAMOS EL ALGROITMO DE LA CUBETA
            #EN LA VARIABLE i GUARDAMOS EL NUMERO DE ORDEN
            #Y EN LA VARIABLE "palabra" GUARDAMOS EL VALOR(STRING)
            for i, palabra in enumerate(palabras):
                #SI EL ELEMENTO "palabraIndividual" COINCIDE CON
                #"palabra" SE AGREGA UN 1 EN ESA POSICIÓN A "cubeta"
                if palabra == palabraIndividual:
                    cubeta[i] = 1

       #/*/*/*/*/RESPUESTA PREDICCIÓN RED NEURONAL/*/*/*/
        resultados = model.predict([np.array(cubeta)])
        #CREAMOS VARIABLES DE INDICES VACÍA DE TIPO NUMPY
        resultadosIndices = np.empty(shape=0)
        #OBTENEMOS EL VALOR MÁS ALTO DE LOS RESULTADOS DE PROBABILIDAD
        valor = np.max(resultados)
        
        #/*/*/*/*/*VALIDACION DE DESCARTE/*//*/*/*//*
        #VALIDAMOS SI HAY UNA PROBABILIDAD DE COINCIDENCIA QUE SEA SUFICIENTE
        #SI ES VERDADERO
        if valor > 0.85 :
            #PASAMOS EL INDICE CON MAS PROBABILIDAD DE "resultados" a "resultadosIndices"
            resultadosIndices = np.argmax(resultados)
        #DE LO CONTRARIO    
        else:
            #ASIGNAMOS DIRECTAMENTE EL INDICE DE TAG DE RESPUESTAS DE DESCARTE
            resultadosIndices = 0
        #ASIGNAMOS A TAGS EL VALOR DE INDICE DE "resultadosIndices"
        tag = tags[resultadosIndices]
        
        #/*/*/*/*/MATCH DE CAETEGORÍA USUARIO-JSON/*/*/*/*/*
        for tagAyu in archivoJson["respuestas"]:
            #VERIFICAMOS SI LA ETIQUETA DETECTADA SE ENCUENTRA EN EL JSON
            if tagAyu["tag"] == tag:
                #SI ES VERDADERO GUARDAMOS EL ARREGLO DE RESPUESTAS
                #DE ESA CATEGORÍA EN LA VARIABLE "respuestas"
                respuesta = tagAyu["respuesta"]
        #SE ELIGE LA RESPUESTA DE FORMA ALEATORIA DEL ARREGLO OBTENIDO
        print("\nUMG: ", random.choice(respuesta),"\n")

#LLAMADA A FUNCION PRINCIPAL
mainChatBot()
