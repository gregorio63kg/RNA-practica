import numpy as np
from FunAct import FuncionesActivacion

class Neurona(FuncionesActivacion):
    def __init__(self, id, tipo, activacion, entradas):
        """
        Inicializa una unidad atómica (Neurona_Lego_Base).
        """
        self.id = id
        self.tipo = tipo
        self.f_activacion = activacion
        self.entradas = np.array(entradas) if entradas is not None else None
        
        # Inicialización de estado interno
        self.pesos = self.iniciar_pesos()
        self.bias = self.iniciar_bias()
        
        # La salida se mantiene en 0 hasta que se procese
        self.salida = 0

    def iniciar_pesos(self):
        if self.entradas is not None:
            # Inicialización aleatoria como define el idea.json
            return np.random.rand(len(self.entradas))
        return None

    def iniciar_bias(self):
        return np.random.random()

    def suma_ponderada(self):
        """Implementación de 'compute' según idea.json"""
        if self.entradas is None or self.pesos is None:
            return 0
        return np.dot(self.entradas, self.pesos) + self.bias

    def procesar(self):
        """
        Ejecuta el ciclo: compute -> activate.
        Retorna el valor de difusión universal (output).
        """
        if self.entradas is not None:
            # Equivale a compute + activate en el JSON
            self.salida = self.activar(self.f_activacion)
        return self.salida

    def mostrar_estado(self):
        """Muestra los datos actuales de la neurona (Ficha técnica)"""
        print(f"--- Ficha Técnica Neurona: {self.id} ---")
        print(f"Tipo        : {self.tipo}")
        print(f"Activación  : {self.f_activacion}")
        print(f"Entradas    : {self.entradas}")
        print(f"Pesos       : {self.pesos}")
        print(f"Bias        : {self.bias}")
        print(f"Salida (Out): {self.salida}")
        print("--------------------------------------\n")

if __name__ == "__main__":
    # --- Prueba de Funcionamiento ---
    
    # Datos de ejemplo
    entradas_ejemplo = [0.8, -0.2, 0.5]
    
    # 1. Crear neurona
    n1 = Neurona(
        id="LEGO-N1", 
        tipo="atomic_unit", 
        activacion="relu", 
        entradas=entradas_ejemplo
    )
    
    # 2. Procesar información
    n1.procesar()
    
    # 3. Mostrar resultado
    n1.mostrar_estado()
    
    # Prueba con Sigmoide
    n2 = Neurona(
        id="LEGO-N2", 
        tipo="atomic_unit", 
        activacion="sigmoide", 
        entradas=entradas_ejemplo
    )
    n2.procesar()
    n2.mostrar_estado()