"""Mi primera red neuronal: aprende la función XOR usando solo NumPy."""

import numpy as np

np.random.seed(42)

#datos entradas y respuestas
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

#funcion activacion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivada(a):
    return a * (1- a)

#parametros 

#arquitectura
#2 entradas 4 neuronas 1 salida
W1 = np.random.rand(2, 4)
b1 = np.random.zeros((1, 4))
W2 = np.random.rand(4, 1)
b2 = np.zeros((1, 1))

tasa_aprendizaje = 0.5

#entrenamiento
for epoca in range(10000):
    #calcular la prediccion
    a1 = sigmoid(X @ W1 + b1)
    a2 = sigmoid(a1 @ W2 +b2)

    #ver la lejania de la respuesta
    perdida   = np.mean((a2 - y) ** 2)

    #contribucion al parametro de error
    d2 = (a2 - y) * sigmoid_derivada(a2)
    dW2 = a1.T @ d2
    db2 = d2.sum(axis=0, keepdims=True)
 
    d1 = (d2 @ W2.T) * sigmoid_derivada(a1)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0, keepdims=True)
 
    #Actualizar: mover los parámetros para reducir el error
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1
 
    if epoca % 1000 == 0:
        print(f"Época {epoca:5d} | pérdida: {perdida:.5f}")
 
#RESULTADOS
print("\nPredicciones finales:")
for entrada, pred in zip(X, a2):
    print(f"{entrada} -> {pred[0]:.3f}")
 
