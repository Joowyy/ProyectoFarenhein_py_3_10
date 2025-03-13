import numpy as np
import tensorflow as tf

# Usamos la libreria de numpy para crear nuestros arreglos (arrays)
celsius = np.array ([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit = np.array ([-40, 14, 32, 46, 59, 72, 100], dtype = float)

# Keras (framework) Unidades son las capas que tenemos
# Dense (tipo de capa) Input_shape (cantidad de neuronas)
capa1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
capa2 = tf.keras.layers.Dense(units = 3)
salida = tf.keras.layers.Dense(units = 1)

# Modelo secuencial, es la forma en que se van apilando las capas
# Es decir, (capa 1, capa 2, capa 3), es un modelo de capas
modelo = tf.keras.Sequential([capa1, capa2, salida])

# Compilador del modelo
modelo.compile(

    # El optimizer Adam, va ajustando el peso y sesgo de la neurona para que aprenda
    # El valor numerico, es la frecuencia con la que se va ejecutando el entrenamiento
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

# Entrenamiento del modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs = 100, verbose = False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show() # Añadido para mostrar el gráfico

# Probar el modelo con algunos valores
print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

# Mostrar los pesos internos del modelo
print("Variables internas del modelo:")
print(capa.get_weights())
