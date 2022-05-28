import nltk as nt
palabras=[]
texto = "Las formas en que uno regularmente busca estar bien, no dependen de nadie externo, aunque en muchos casos se asociona la idea erronea de que la felicidad está ligada a una relación de pareja"

stemmer = nt.stem.lancaster.LancasterStemmer()   

salida = nt.word_tokenize(texto)

print("Tokens:  ",salida)
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
 
salida = [stemmer.stem(p.lower()) for p in salida if p != "?"] 

print("RAICES DE LAS PALABRAS: ",salida)