---
type: concept
status: stable
tags: [architecture, sequence-flow, object-oriented]
code_source: neurona.py
---

# Arquitectura y Flujo de Procesamiento

Este documento describe la relación estructural de las clases y el orden cronológico en el que se ejecutan los cálculos al procesar entradas en la neurona.

---

## 🏗️ Jerarquía de Clases (Herencia)

El diseño del proyecto utiliza herencia para separar la lógica matemática de activación del estado e inicialización de la neurona:

```mermaid
classDiagram
    class FuncionesActivacion {
        +normalizacion() Array
        +tang() float
        +sigmoide() float
        +relu() float
        +activar(nombre_funcion) float
    }
    class Neurona {
        +id: String
        +tipo: String
        +f_activacion: String
        +entradas: Array
        +pesos: Array
        +bias: float
        +salida: float
        +iniciar_pesos() Array
        +iniciar_bias() float
        +suma_ponderada() float
        +procesar() float
        +mostrar_estado() void
    }
    FuncionesActivacion <|-- Neurona : Hereda
```

*Nota: [Neurona](../../neurona.py) hereda directamente de [FuncionesActivacion](../../FunAct.py), permitiéndole invocar métodos como `self.activar()` y acceder a las funciones individuales de activación de manera directa.*

---

## 🔄 Flujo de Procesamiento de una Neurona

Cuando se invoca el método `procesar()` en una instancia de `Neurona`, se inicia una secuencia de llamadas internas estructurada de la siguiente manera:

```mermaid
sequenceDiagram
    autonumber
    actor Cliente as Programa Cliente
    participant N as Instancia Neurona
    participant FA as Clase Base (FuncionesActivacion)

    Cliente->>N: n1.procesar()
    alt Entradas no nulas
        N->>FA: self.activar(f_activacion)
        Note over FA: Resuelve dinámicamente el método con getattr()
        FA->>N: Invoca la función específica (e.g., self.sigmoide())
        N->>N: self.suma_ponderada()
        Note over N: z = np.dot(entradas, pesos) + bias
        N-->>FA: Retorna suma ponderada (z)
        Note over FA: Aplica fórmula de activación a z
        FA-->>N: Retorna valor activado (a)
        Note over N: Guarda resultado en self.salida
    else Entradas nulas
        Note over N: Asigna self.salida = 0
    end
    N-->>Cliente: Retorna self.salida
```

### Explicación del Ciclo:
1. **Llamada a `procesar()`**: El programa cliente inicializa las entradas y solicita procesar.
2. **Despacho de Activación**: `procesar()` delega en `activar(...)` de la clase base.
3. **Llamada de Vuelta (Callback)**: El método de activación seleccionado (como `sigmoide()`, `tang()`, `relu()`) llama a `self.suma_ponderada()` para obtener la entrada neta combinada.
4. **Cálculo de la Suma Ponderada**: `suma_ponderada()` ejecuta la suma ponderada lineal.
5. **Aplicación de No-Linealidad**: La función de activación aplica su fórmula sobre la suma ponderada.
6. **Guardado e Interconexión**: La neurona almacena su salida activada, lista para ser leída por otras neuronas.
