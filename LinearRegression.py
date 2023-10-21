import numpy as np

class Linear_regression_vectorized():

    def __init__(self):
        self.__weights = 0
        self.__bias = 0
        self.__mu = 0
        self.__sigma = 0

    def get_weights(self):
        return self.__weights

    def get_bias(self):
        return self.__bias

    def get_mean(self):
        return self.__mu

    def get_std(self):
        return self.__sigma


    def compute_cost(self, X, y, w, b):
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
        

    def __compute_gradients(self, X, y, w, b):
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


    
    def run_gradient_descent(self, X, y, w, b, alpha=0.01, num_iter=10, verbose=False):
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
            dj_dw, dj_db = self.__compute_gradients(X, y, w, b)
            
            # model params update
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            # compute cost
            cost = self.compute_cost(X, y, w, b)
            # append cost and params
            history_cost[i] = cost
            history_params[i] = np.concatenate((w, np.array([b])))
            
            # print
            if verbose:
                import math
                if i% math.ceil(num_iter/10) == 0:
                    print(f"Costo hasta la iteración {i}: {cost}")
        
        self.__weights = w
        self.__bias = b
        return w, b, history_cost, history_params


    def z_scaling_feature(self, X):
        """
        Aplica la normalización z-score sobre la data proporcionada.

        Args:
            - X (ndarray (m,n)): dataset.

        Returns:
            - X_norm (ndarray (m,n)): dataset normalizada.
        """

        self.__mu = np.mean(X, axis=0) # (n,)
        self.__sigma = np.std(X, axis=0) # (n, )

        X_norm = (X - self.__mu) / self.__sigma
        return X_norm



if __name__ == "__main__":
    ## test ##

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

    model = linear_regression_vectorized()

    w_final, b_final, history_cost, history_params = model.run_gradient_descent(X, y, w, b, alpha, num_iter, True)

    print(f"w final: {w_final}\nb final:{b_final}")


    # prediction
    print(f"x = 1000, y_prediction = {w_final * 1.0 + b_final}")
    print(f"x = 1200, y_prediction = {w_final * 1.2 + b_final}")
    print(f"x = 2000, y_prediction = {w_final * 2.0 + b_final}")


    # grafica de prediction
    import matplotlib.pyplot as plt

    plt.scatter(X, y, c="r")
    plt.plot(X, X@model.get_weights()+model.get_bias(), c="b")
    plt.show()


    # grafica iterations vs cost
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(range(num_iter)[:100], history_cost[:100])
    ax2.plot(range(num_iter)[1000:], history_cost[1000:])
    ax1.set_title("Núm. iteraciones vs costo (inicio)")
    ax2.set_title("Núm. iteraciones vs costo (final)")
    ax1.set_xlabel("Num. iteraciones")
    ax1.set_ylabel("Costo")
    ax2.set_xlabel("Num. iteraciones")
    ax2.set_ylabel("Costo")
    plt.show()


    # testing z-score normalization
    # peak to peak mide el rango de un array: max_value - min_value
    X_norm= model.z_scaling_feature(X)
    X_mu = model.get_mean()
    X_sigma = model.get_std()
    print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")



