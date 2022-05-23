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


 #CREAMOS UN OBJETO DE LA CLASE lancasterStemmer
 #CON SU METODO stem NOS PERMITE OBTENER LA RAÍZ LEXICA DE UNA PALABRA
stemmer = nt.stem.lancaster.LancasterStemmer()


#OBTENEMOS LOS DATOS DE NUESTRO ARCHIVO JSON
with open("respuestas.json") as f:
    #VOLCAMOS LOS DATOS DEL ARCHIVO JSON EN UNA VARIABLE
    contenido = json.load(f)

#LISTA QUE CONTIENE LAS PALABRAS RECONOCIDAS
palabras = []
#LISTA QUE CONTIENE LAS ETIQUETAS DE LAS PALABRAS
tags = []
#
auxX=[]
auxY=[]
#RECORREMOS EL ARCHIVO JSON EN EL ARREGLO DE CATEGORÍAS
for i in contenido["respuestas"]:
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

print("Palabras: ",palabras)
print("Auxiliar en X: ",auxX)
print("Auxiliar en Y: ",auxY)
print("Tags: ",tags)

#EL ARREGLO PALABRAS CONTIENE LOS TOKENS Y BUSCAMOS LAS RAÍCES DE LAS PALABRAS 
#Y EVITAMOS LOS SIGNOS DE INTERROGACIÓN
palabras = [stemmer.stem(p.lower()) for p in palabras if p != "?"] 
#DEVOLVEMOS UNA LISTA ORDENADA DE palabras
palabras = sorted(list(set(palabras)))
#DEVOLVEMOS UNA LISTA ORDENADA DE tags
tags = sorted(tags)

#CREAMOS LISTA PARA ENETRENAMIENTO DE LA RED NEURONAL
entrenamiento = []
#CREAMOS LISTA PARA LOS DATOS DE SALIDA
salida = []

#CREAMOS UNA LISTA LLENA DE LA CANTIDAD DE CEROS COMO LA LONGITUD DE LA LISTA tags
salidaVacia = [0 for _ in range(len(tags))]

#APLICACIÓN DEL ALGORITMO DE LA CUBETA
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


print(entrenamiento)
print(salida)

#CONVERTIMOS LAS 2 LISTAS EN ARRAYS DE NUMPY
entrenamiento = np.array(entrenamiento)
salida = np.array(salida)

#METODO QUE NOS SIRVE PARA LIMPIAR NUESTRO ESPACIO DE TRABAJO
tf.compat.v1.reset_default_graph()

net = tl.input_data(shape = [None, len(entrenamiento[0]) ])
net = tl.fully_connected(net, 10)
net = tl.fully_connected(net, 10)
net = tl.fully_connected(net, len(salida[0]), activation = "softmax")
net = tl.regression(net)

model = tl.DNN(net)
model.fit(entrenamiento, salida, n_epoch = 1000, batch_size = 10, show_metric= True)
model.save("modelo.tflearn")

def mainChatBot():
    while True:
        entrada = input("Tu:")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProc = nt.word_tokenize(entrada)
        entradaProc

