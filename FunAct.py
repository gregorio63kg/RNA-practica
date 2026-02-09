def normalizacion(self):
    return (self.entradas - np.min(self.entradas)) / (np.max(self.entradas) - np.min(self.entradas))
def tang(self):
    return np.tanh(self.suma_ponderada())

def sigmoide(self):
    return 1 / (1 + np.exp(-(self.suma_ponderada())))

def relu(self):
    return np.maximum(0,self.suma_ponderada()*0.5)

def entrada(self):
    return normalizacion(self.entradas)

def activar(self, nombre_funcion):
    metodo = getattr(self, nombre_funcion, None)
    if metodo and callable(metodo):
        return metodo()
    raise ValueError(f"Error: '{nombre_funcion}' no es una función de activación válida.")
