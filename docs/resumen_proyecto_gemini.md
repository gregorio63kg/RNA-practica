# Resumen de Proyecto y Estado de Desarrollo: RNA-practica

Este documento contiene un resumen completo del estado actual del proyecto `RNA-practica` estructurado bajo el método **LLM Wiki (Karpathy)** y las especificaciones de **habilidades para agentes de Obsidian**. Está diseñado para que otro modelo (como Gemini) entienda rápidamente el contexto técnico y continúe con el desarrollo.

---

## 1. Estructura de Archivos Actual (Workspace)

```text
G:.
│   .gitignore
│   FunAct.py                  <-- Definición de funciones de activación y normalización
│   neurona.py                 <-- Script principal y punto de entrada (Clase Neurona)
│   CLAUDE.md                  <-- Manifiesto / Guía de estilo del Compilador LLM Wiki
│
├───docs
│   ├───raw                    <-- Capa 1: Fuentes crudas e inmutables
│   │       idea.json          <-- Especificación de diseño original de la neurona
│   │       llm-wiki.md        <-- Especificación original del Método Karpathy
│   │
│   └───wiki                   <-- Capa 2: Wiki sintetizado por el LLM (Notas de Obsidian)
│           Indice MOC.md      <-- Mapa de contenidos y punto de entrada visual
│           Neurona Lego Base.md
│           Funciones de Activacion.md
│           Arquitectura y Flujo.md
│           Mapa Neuronal.canvas <-- Mapa espacial interactivo de Obsidian
│
└───.claude                    <-- Habilidades de Obsidian (kepano/obsidian-skills)
    │   LICENSE
    │   README.md
    │
    ├───.claude-plugin
    │       marketplace.json
    │       plugin.json        <-- Manifiesto de la extensión de habilidades
    │
    └───skills                 <-- Instrucciones de formato para el agente de IA
        ├───defuddle
        ├───json-canvas
        ├───obsidian-bases
        ├───obsidian-cli
        └───obsidian-markdown
```

---

## 2. Archivo de Configuración / Esquema Central

### A. Manifiesto del Compilador (`CLAUDE.md`)
Este archivo sirve como las directrices para que los agentes de IA mantengan el orden y estilo de la documentación:

```markdown
# Compilador LLM Wiki (Método Karpathy) - CLAUDE.md

Este archivo contiene las directrices de estilo y las reglas de compilación para los agentes de Inteligencia Artificial que mantienen esta base de conocimiento. Al actuar en este repositorio, debes operar bajo el **Método Karpathy (LLM Wiki)**.

- **Capa 1: Fuentes Crudas (`docs/raw/`)**: Documentos originales (e.g. `idea.json`), registros de chat y transcripciones (inmutables).
- **Capa 2: El Wiki Sintetizado (`docs/wiki/`)**: Notas de Markdown interconectadas que representan la síntesis del código fuente.
- **Capa 3: El Esquema / Reglas (`CLAUDE.md`)**: Reglas de consistencia que rigen la documentación.
```

### B. Especificación de Diseño Original (`docs/raw/idea.json`)
```json
{
    "component_name": "Neurona_Lego_Base",
    "version": "1.0.0",
    "type": "atomic_unit",
    "description": "Unidad de procesamiento independiente con conectividad dinámica y autogestión de pesos.",
    "structure": {
        "inputs": {
            "type": "dynamic_map",
            "properties": { "source_id": "string", "value": "float", "weight": "float" }
        },
        "internal_state": {
            "bias": { "type": "float", "initialization": "random" },
            "activation_function": { "type": "enum", "options": ["relu", "sigmoid", "tanh", "linear"] }
        },
        "output": { "type": "float" }
    },
    "methods": {
        "compute": "sum(inputs[i].value * inputs[i].weight) + bias",
        "activate": "apply(activation_function, compute_result)",
        "update": "weight = weight - (learning_rate * error_buffer * input_value)"
    }
}
```

---

## 3. Script Principal / Punto de Entrada

### `neurona.py`
Inicializa la neurona utilizando las funciones heredadas de `FunAct.py` y corre una prueba unitaria:

```python
import numpy as np
from FunAct import FuncionesActivacion

class Neurona(FuncionesActivacion):
    def __init__(self, id, tipo, activacion, entradas):
        self.id = id
        self.tipo = tipo
        self.f_activacion = activacion
        self.entradas = np.array(entradas) if entradas is not None else None
        self.pesos = self.iniciar_pesos()
        self.bias = self.iniciar_bias()
        self.salida = 0

    def iniciar_pesos(self):
        if self.entradas is not None:
            return np.random.rand(len(self.entradas))
        return None

    def iniciar_bias(self):
        return np.random.random()

    def suma_ponderada(self):
        if self.entradas is None or self.pesos is None:
            return None
        return np.dot(self.entradas, self.pesos) + self.bias

    def procesar(self):
        if self.entradas is not None:
            self.salida = self.activar(self.f_activacion)
        else:
            self.salida = 0
        return self.salida

    def mostrar_estado(self):
        print(f"--- Ficha Técnica Neurona: {self.id} ---")
        print(f"Tipo        : {self.tipo}")
        print(f"Activación  : {self.f_activacion}")
        print(f"Entradas    : {self.entradas}")
        print(f"Pesos       : {self.pesos}")
        print(f"Bias        : {self.bias}")
        print(f"Salida (Out): {self.salida}")
        print("--------------------------------------\n")

if __name__ == "__main__":
    entradas_ejemplo = [0.8, -0.2, 0.5, 0.1]
    n1 = Neurona(id="LEGO-N1", tipo="atomic_unit", activacion="sigmoide", entradas=entradas_ejemplo)
    n1.procesar()
    n1.mostrar_estado()
```

---

## 4. Flujo Actual y Bloqueo / Siguientes Pasos

- **Lo que funciona actualmente**: 
  El proyecto ejecuta cálculos aislados de una neurona individual combinando suma ponderada y funciones de activación dinámica. La bóveda de Obsidian se ha reestructurado completamente bajo el Método Karpathy (`raw/` y `wiki/`) y las habilidades del agente están instaladas en `.claude/`.
- **Objetivo / Problema inmediato**: 
  Se debe evolucionar el script para admitir conectividad dinámica (red de neuronas interconectadas en vez de entradas fijas en el constructor) e implementar la retropropagación en el método de actualización (`update`), aplicando la regla de aprendizaje de pesos utilizando el búfer de error local de cada neurona.
