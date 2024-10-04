import numpy as np
import matplotlib.pyplot as plt

# Definición del operador O
oOper = np.array([[0, 1], [1, 0]])

# Estado inicial
yInit = np.array([[1, 0], [0, 0]])

# Definición del tiempo y tamaño del paso
times = np.linspace(0.0, 10.0, num=50)
h = times[1] - times[0]

# Deep copy del estado inicial
yCopy = yInit.copy()

# Definición de las funciones necesarias
def dyn_generator(t, O, y):
    """Generador de dinámica f(t, y) = -i[O, y]"""
    return -1j * (np.dot(O, y) - np.dot(y, O))

def rk4(f, O, y, h):
    """Método RK4 para resolver ecuaciones diferenciales ordinarias."""
    k1 = h * f(0, O, y)
    k2 = h * f(0, O, y + 0.5 * k1)
    k3 = h * f(0, O, y + 0.5 * k2)
    k4 = h * f(0, O, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Inicialización de los arreglos para guardar los resultados
stateQuant00 = np.zeros(times.size)
stateQuant11 = np.zeros(times.size)

# Evolución temporal
for tt in range(times.size):
    stateQuant00[tt] = yInit[0, 0].real
    stateQuant11[tt] = yInit[1, 1].real

    # Aplicar método RK4 para evolucionar el sistema
    yInit = rk4(dyn_generator, oOper, yInit, h)

# Graficar los resultados
plt.style.use('_mpl-gallery')

x = times
y = stateQuant00
y2 = stateQuant11

plt.plot(x, y, 'g.-', linewidth=1.5)
plt.plot(x, y2, '.-', linewidth=1.5)

plt.title("Evolución temporal de las entradas (0,0) y (1,1) del estado estudiado.")
plt.xlabel("Tiempo")
plt.ylabel("Entradas del estado")
plt.legend(["Entrada (0,0)", "Entrada (1,1)"], loc="upper left", bbox_to_anchor=(1.05, 0.75), ncol=1)

# Guardar la figura
plt.savefig('Evolucion_de_las_entradas.png')

# Mostrar la figura
plt.show()
