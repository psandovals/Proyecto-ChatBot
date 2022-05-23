import nltk as nt

texto = "Las formas en que uno regularmente busca estar bien, no dependen de nadie, aunque en muchos casos se asociona la idea erronea de que la felicidad está ligada a una relación de pareja"

salida = nt.word_tokenize(texto)

print(salida)