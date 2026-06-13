---
type: theory
status: stable
tags: [neural-networks, math, activations]
code_source: FunAct.py
---

# Funciones de Activación

Las funciones de activación determinan si una neurona debe activarse o no a partir del cálculo de su suma ponderada. Aportan **no linealidad** al modelo, permitiendo que la red neuronal aprenda relaciones complejas en los datos.

En este proyecto, se definen en la clase `FuncionesActivacion` dentro de [FunAct.py](../../FunAct.py).

---

## 📈 Funciones Implementadas

### 1. Tangente Hiperbólica (Tanh)
La función Tanh escala el resultado del procesamiento en un rango de entre **-1 y 1**. Es útil para centrar los datos alrededor de cero.

#### Ecuación Matemática:
$$f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

#### Implementación en Código:
```python
def tang(self):
    return np.tanh(self.suma_ponderada())
```

---

### 2. Sigmoide (Logistic)
La función Sigmoide mapea cualquier valor real al rango **(0, 1)**. Se utiliza habitualmente en la capa de salida para problemas de clasificación binaria.

#### Ecuación Matemática:
$$f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Implementación en Código:
```python
def sigmoide(self):
    return 1 / (1 + np.exp(-(self.suma_ponderada())))
```

---

### 3. Rectified Linear Unit modificada (ReLU Modificada)
La ReLU tradicional devuelve 0 si la entrada es negativa, y la misma entrada si es positiva ($f(z) = \max(0, z)$). En este proyecto se utiliza una variante modificada que multiplica la parte positiva por $0.5$.

#### Ecuación Matemática:
$$f(z) = \max(0, z \cdot 0.5)$$

#### Implementación en Código:
```python
def relu(self):
    return np.maximum(0, self.suma_ponderada() * 0.5)
```

---

## ⚖️ Normalización Min-Max

Adicionalmente, se incluye un método para normalizar los datos de entrada a una escala uniforme de entre **0 y 1**.

#### Ecuación Matemática:
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

#### Implementación en Código:
```python
def normalizacion(self):
    return (self.entradas - np.min(self.entradas)) / (np.max(self.entradas) - np.min(self.entradas))
```

---

## 🛠️ Despacho Dinámico de Funciones

La clase implementa un despachador dinámico utilizando la función nativa de Python `getattr`. Esto permite llamar a cualquiera de las funciones de activación pasando simplemente una cadena de texto (e.g., `"sigmoide"`):

```python
def activar(self, nombre_funcion):
    metodo = getattr(self, nombre_funcion, None)
    if metodo and callable(metodo):
        return metodo()
    raise ValueError(f"Error: '{nombre_funcion}' no es una función de activación válida.")
```

De esta forma, en [[Neurona Lego Base]] se ejecuta dinámicamente el ciclo de activación según el estado `self.f_activacion`.
