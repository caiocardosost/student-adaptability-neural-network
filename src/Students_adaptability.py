# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:53:58 2026

@author: caioc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""--------------------Parte 1 - Preparando o dataset------------------------"""

df = pd.read_csv("data/students_adaptability.csv")

X_data = df.drop(columns = ["Adaptivity Level"]) #Selecionando colunas para X data  
Y_data = df["Adaptivity Level"] #Selecionando coluna do Y_data (alvo) - Caso use a forma indireta de one hot encode, essa linha nao existe


"""--------------------Parte 1.1 - Encoding das variaveis nominais e ordinais------------------------"""

# Variáveis Nominais com One-Hot Encoding


#Education Level Institution (Tipo de instituição de ensino)
X_data = pd.get_dummies(X_data,columns=["Education Level"], prefix="", prefix_sep="", dtype=int)

# Device
X_data = pd.get_dummies(X_data,columns=["Device"], prefix="", prefix_sep="", dtype=int)

#Variáveis Binárias (0/1)

#Genero
X_data["Gender"] = X_data["Gender"].map({"Boy": 0, "Girl": 1})

#Education Institution type Não governamental(0), Governamental(1)
X_data["Institution Type"] = X_data["Institution Type"].map({"Non Government": 0, "Government": 1})

#Estudante de TI Não(0), Sim(1)
X_data["IT Student"] = X_data["IT Student"].map({"No": 0, "Yes": 1})

#Localiza-se na Cidade Não(0), Sim(1)
X_data["Location"] = X_data["Location"].map({"No": 0, "Yes": 1})

#Load shedding Baixo(0), Alto(1)
X_data["Load-shedding"] = X_data["Load-shedding"].map({"Low": 0, "High": 1})

#Internet type Mobile Data(0), Wifi(1)
X_data["Internet Type"] = X_data["Internet Type"].map({"Mobile Data": 0, "Wifi": 1})

#LMS Não(0), Sim(1)
X_data["Self Lms"] = X_data["Self Lms"].map({"No": 0, "Yes": 1})

# Variáveis Ordinais: (escala definida)
# Mapeia 'Financial Condition' para valores ordinais: Poor (0), Mid (1), Rich (2)
X_data["Financial Condition"] = X_data["Financial Condition"].map({"Poor": 0, "Mid": 1, "Rich": 2})

# Mapeia 'Network Type' para valores ordinais: 2G (0), 3G (1), 4G (2)
X_data["Network Type"] = X_data["Network Type"].map({"2G": 0, "3G": 1, "4G": 2})

# Encoding de classes representadas em intervalos
faixas_idade = {
    "1-5": 3,
    "6-10": 8,
    "11-15": 13,
    "16-20": 18,
    "21-25": 23,
    "26-30": 28,
    "31-35": 33,
    "36-40": 38
}
duracao = {
    "0": 0,
    "1-3": 2,
    "3-6": 4.5
}
# Realizando o mapeamento
X_data["Age"] = X_data["Age"].map(faixas_idade)
X_data["Class Duration"] = X_data["Class Duration"].map(duracao)

# Mapeia 'Adaptivity Level' para valores ordinais: Low (0), Moderate (1), High (2)
Y_data = pd.get_dummies(Y_data,columns=["Adaptivity Level"], prefix="", prefix_sep="", dtype=int)
Y_data = Y_data.to_numpy() #transformando em matriz (simplifica)

"""--------------------Parte 2 - Preparando a rede neural------------------------"""
# criando nossa rede neural - NN - neural network:
# 3 Neuronios - Logo temos "3" linhas na matriz de pesos
# 18 colunas - 17 caracteristicas (colunas do X_data) + linha do bias
# inicializa com pesos aleatorios - o "-0.5" é para normalizar entre -1,0 e 1

NN = np.random.rand(3,X_data.shape[1]+1) - 0.5

#adicionando o "1" do Bias nas entradas:
X_data = np.append(X_data, np.ones((X_data.shape[0],1)), axis = 1)

#funcao de ativacao sigmoid
def activation(x):
    return 1 / (1 + np.exp(-x))

# Funcao que calcula Y"chapeu" - Y = activaction(S) e S = X*WT - ou seja: S = Sum(Xi*WiT)
# abaixo eh so um teste, pos essa funcao eh aplicada no treinamento
#Y_out = activation(np.matmul(X_data[2,:],np.transpose(NN)))  #matmul retorna um vetor e actiovation aplica a funcao em cada elemento do vetor

"""--------------------Parte 3 - Treinando a rede neural------------------------"""
train = int (0.8*X_data.shape[0]) #80% treino
test = int (X_data.shape[0] - train) #20% teste
print(f'Total: {X_data.shape[0]}, Train: {train}, Test: {test}')

X_train = X_data[0:train,:]
X_test = X_data[train:,:]
Y_train = Y_data[0:train]
Y_test = Y_data[train:]

#numero de epocas
epochs = 100
learning_rate = 0.01


for epoch in range(epochs):
    for i in range(len(X_train)):
        input_line = X_train[i,:] #pega a entrada atual
        Y_out = activation(np.matmul(input_line,np.transpose(NN))) #calcula Y "chapeu"
        error = Y_out - Y_train[i,:] #Verifica o erro: aqui seria utilizado erro quadratico, mas como a derivada perde o quadrado, omitimos
    #Backpropagation: Altera todos os pesos da NN em atraves do gradiente descendente derivada do erro em funcao dos pesos
        for i in range(NN.shape[0]):
            for j in range (NN.shape[1]):
                # Wij = - Wij*2*Learning_rate*Derror/DWij, onde Derror/DWij = 2*(Y_out - Y_train)  Y_out*(1-Y_out)*Xj
                NN[i,j] -= learning_rate * 2* error[i] * Y_out[i]*(1-Y_out[i])*input_line[j]

"""--------------------Parte 3 - Testando a rede neural treinada------------------------"""

for i in range(10): #10 testes
    input_line = X_test[i,:] #pegando a entrada atual
    Y_out = activation(np.matmul(input_line,np.transpose(NN))) # obtem Y Chapeu    
    prediction = np.zeros(3) #controi o vetor de predicao para a entrada atual
    prediction[np.argmax(Y_out)] = 1 # Consulta qual posicao de Y_out tem maior valor e assim, em prediction, muda o valor nesta posicao para 1
    prediction = prediction.astype(int) #Transformando em int (visualizacao)
    print(Y_test[i,:])#comparando resultados
    print(prediction)#comparando resultados
    print()