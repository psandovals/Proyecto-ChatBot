#LIBRERIAS UTILIZADAS
import nltk as nt
import tensorflow as tf
import tflearn as tl
import numpy as np 
import json 
import random 
import pickle
 
 #CONVERTIR MÁS LEGIBLE NUESTRO CHAT
    #stemmer = nt.stem.lancaster.LancasterStemmer()


#OBTENEMOS LOS DATOS DE NUESTRO ARCHIVO JSON
with open("respuestas.json") as f:
    contenido = json.load(f)

#IMPRIMIMOS EN PANTALLA LA CARGA DEL ARCHIVO JSON
print(contenido)