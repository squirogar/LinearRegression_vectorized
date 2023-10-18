import numpy as np

def compute_cost(X, y, w, b):
    """
    Mide el desempeño del modelo de regresión lineal calculando el error cuadrático 
    medio (Mean Squared Error).

    Args:
        - X (ndarray (m,n)): dataset. 
        - y (ndarray (m,)): etiqueta o salida verdadera.
        - w (ndarray (n,)): pesos o weights.
        - b (float): sesgo o bias.
        Nota: 'm' es el número de ejemplos y 'n' es el número de features.
        
    Returns:
        - cost (float): costo o error cuadrático medio.
    """
    
    m = X.shape[0]
    prediction = np.dot(X, w) + b
    sum_errors = np.sum((prediction - y) ** 2)
    cost = sum_errors / (2 * m)
    
    return cost
        

def compute_gradients(X, y, w, b):
    """
    Calcula las derivadas parciales para todos los weights y bias del modelo de
    regresión lineal.
    
    Args:
        - X (ndarray (m,n)): dataset. 
        - y (ndarray (m,)): etiqueta o salida verdadera.
        - w (ndarray (n,)): pesos o weights.
        - b (float): sesgo o bias.
        Nota: 'm' es el número de ejemplos y 'n' es el número de features.
    
    Returns:
        - dj_dw (ndarray (n,)): numpy array de derivadas parciales de la función
                                de costo con respecto a los weigths.
        - dj_db (float): derivada parcial de la función de costo con respecto al
                         bias.
    """
    m = X.shape[0]
    prediction = np.dot(X, w) + b
    error = prediction - y
    dj_dw = np.dot(error, X) / m
    dj_db = np.mean(error)
    
    return dj_dw, dj_db


    
def run_gradient_descent(X, y, w, b, alpha, num_iter, verbose=False):
    """
    Ejecuta Batch Gradient Descent para entrenar el modelo de regresión lineal.
    
    Args:
        - X (ndarray (m,n)): dataset. 
        - y (ndarray (m,)): etiqueta o salida verdadera.
        - w (ndarray (n,)): pesos o weights.
        - b (float): sesgo o bias.
        - alpha (float): learning rate.
        - num_iter (int): número de iteraciones o epochs del gradient descent.
        - verbose (bool): True para imprimir el costo mientras se ejecuta el algoritmo.
        Nota: 'm' es el número de ejemplos y 'n' es el número de features.    
    
    Returns:
        - w (ndarray (n,)): pesos o weights finales.
        - b (int): sesgo o bias final.
        - history_cost (ndarray (num_iter,)): historial del costo obtenido al evaluar el modelo.
        - history_params (ndarray (num_iter,n+1)): historial de parámetros del modelo calculados 
                                                   por el algoritmo.
    """
    history_cost = np.zeros((num_iter,))
    history_params = np.zeros((num_iter, w.shape[0] + 1))
    
    for i in range(num_iter):
        # gradients
        dj_dw, dj_db = compute_gradients(X, y, w, b)
        
        # model params update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # compute cost
        cost = compute_cost(X, y, w, b)
        # append cost and params
        history_cost[i] = cost
        history_params[i] = np.concatenate((w, np.array([b])))
        
        # print
        if verbose:
            import math
            if i% math.ceil(num_iter/10) == 0:
                print(f"Costo hasta la iteración {i}: {cost}")
    
    return w, b, history_cost, history_params


# Load our data set
X = np.array([[1.0], [2.0]])   #features
y = np.array([300.0, 500.0])   #target value


# initialize parameters
w = np.array([0])
print(w.shape)
b = 0
# some gradient descent settings
num_iter = 10000
alpha = 1.0e-2


w_final, b_final, history_cost, history_params = run_gradient_descent(X, y, w, b, alpha, num_iter, True)

print(f"w final: {w_final}\nb final:{b_final}")
