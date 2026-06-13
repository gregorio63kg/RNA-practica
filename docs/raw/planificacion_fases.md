# Planificación de Fases - Roadmap de RNA-practica

Este documento contiene la planificación del desarrollo del proyecto `RNA-practica` para evolucionar la unidad atómica de la neurona a una red interconectada.

---

## 📅 Fase 1: Conectividad Dinámica (Rediseño de Entradas)
- **Objetivo**: Reemplazar las entradas estáticas (`np.array`) por un mapa dinámico (`dict`) mapeado por UUID u origen de la neurona, tal como se especifica en `idea.json`.
- **Tareas**:
  - [ ] Modificar constructor de `Neurona` para admitir diccionario de entradas.
  - [ ] Implementar un método para conectar salidas de neuronas anteriores a las entradas de la actual.
  - [ ] Adaptar la función `suma_ponderada` al nuevo mapa dinámico.

## 📅 Fase 2: Retropropagación y Aprendizaje (Método Update)
- **Objetivo**: Implementar el cálculo del error y la actualización de los pesos a través del método `update`.
- **Tareas**:
  - [ ] Añadir atributo `error_buffer` (delta) a `Neurona`.
  - [ ] Implementar cálculo de la derivada de cada función de activación en `FunAct.py`.
  - [ ] Escribir el método `update(learning_rate, error_buffer, input_value)` en `Neurona`.

## 📅 Fase 3: Orquestador de Red (Lego Connectors)
- **Objetivo**: Crear una clase `RedLego` para conectar múltiples neuronas en capas sin requerir un orquestador central rígido.
- **Tareas**:
  - [ ] Definir clase `RedLego`.
  - [ ] Implementar métodos para añadir capas y mapear conexiones dinámicas automáticamente.
  - [ ] Probar flujo de información directo (forward pass) a través de la red.

## 📅 Fase 4: Entrenamiento y Simulación (Caso XOR / AND)
- **Objetivo**: Validar el sistema completo entrenando la red modular para resolver un problema lineal (AND) y uno no lineal (XOR).
- **Tareas**:
  - [ ] Diseñar el script de entrenamiento.
  - [ ] Registrar métricas de pérdida (loss) por época.
  - [ ] Visualizar los resultados en una nota de la wiki.
