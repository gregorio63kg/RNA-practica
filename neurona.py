import numpy as np


class Neurona:
    
    def __init__(self, id, tipo, estado, activacion, entradas):
        self.id = id
        self.estado = estado # true activo , false  desconetada
        self.tipo = tipo
        self.f_activacion = activacion
    # ________entadas, pesos  de logitud de datos iguales________
        self.entradas = entradas 
        self.pesos =  self.Pesos() 
     #_________inicializar bias__________
        self.bias = self.iniciar_bias() 
    #_________salida de la neurona________    
        self.salida = self.salida_n() if self.entradas != None else 0 
        
   #para inicialiar los peso
    def Pesos(self):
       
        t_entrada = len(self.entradas)
        self.pesos = np.random.rand(1,t_entrada)[0]
        return self.pesos

  #inicializar bias      
    def iniciar_bias(self):
        return np.random.random()

    #suma ponderada     
    def suma_ponderada(self):
     z = np.dot(self.entradas,self.pesos)
     z = z.sum()+ self.bias 
     return z

   #ejecutar la funcion de activacion     
    def salida_n(self):
        return self.f_activacion(self.suma_ponderada())
          
   #muestras los datos de la neurona         
    def Estado(self):
        "muestas los datos de la neurona"
        #print("Estado      :", self.estado)
        print("Neurona     :", self.id )
        #print("Tipo        :", self.tipo)
        print("Act_Selec   :", self.f_activacion)
        print("Entradas    :", self.entradas)
        print("Pesos       :", self.pesos)
        print("Bias        :", self.bias)
        print("Salida      :", self.salida)
        print("\n")

    
        
if __name__ == "__main__":
    
    