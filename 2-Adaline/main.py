# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:35:35 2017

@author: Waldo Hasperué
"""
#%%
from adaline  import train
from ImagenPuntosMPL import AbrirImagen

#%%
root_path = "D:\\DataMining\\Redes Neuronales\\redes\\2-Adaline\\"

file_path = root_path  + "Imagen 1.bmp"

#%%
X = AbrirImagen(file_path)
column_count = 3

T = X[:, column_count - 1]
P = X[:, 0:(column_count - 1)]

alfa = 0.5
max_ite = 50
cota = 0.00001

funcion = 'tansig'
T = T * 2 - 1

#%%
xmax = P.max()
xmin = P.min()
P = (P - xmin)/(xmax - xmin)

#%%
(W, b, ite, error_prom) = train(P, T, alfa, max_ite, cota, funcion, True)



# %%
