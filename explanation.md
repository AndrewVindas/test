# Método Numérico Runge-Kutta de Orden 4

En el campo del análisis numérico, los métodos de Runge-Kutta forman un conjunto de técnicas iterativas, tanto explícitas como implícitas, utilizadas para resolver ecuaciones diferenciales ordinarias, en particular, problemas con condiciones iniciales. Este conjunto de métodos fue desarrollado a principios del siglo XX por los matemáticos alemanes Carl Runge y Martin Kutta.

En esta documentación, nos enfocaremos en el método de Runge-Kutta de cuarto orden (RK4). Consideremos un problema de la forma:

$$
\frac{dy}{dt} = f(t, y)
$$

sujeto a la condición inicial:

$$
y(t_0) = y_0.
$$

El objetivo es determinar el estado futuro \( y_{n+1} \) a partir del estado actual \( y_n \) mediante un proceso iterativo:

$$
y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

Donde los coeficientes \( k_1, k_2, k_3, k_4 \) son:

\[
\begin{aligned}
k_1 & = h \cdot f(t_n, y_n) \\
k_2 & = h \cdot f\left(t_n + \frac{h}{2}, y_n + \frac{k_1}{2}\right) \\
k_3 & = h \cdot f\left(t_n + \frac{h}{2}, y_n + \frac{k_2}{2}\right) \\
k_4 & = h \cdot f(t_n + h, y_n + k_3)
\end{aligned}
\]

Aquí, \( h \) es el tamaño del paso, es decir, el incremento \( \Delta t_n \) entre los puntos sucesivos \( t_n \) y \( t_{n+1} \).

## Implementación del Método RK4

El siguiente código en Python ilustra cómo aplicar el método RK4 para resolver una ecuación diferencial ordinaria:

```python
import numpy as np

def rk4_step(f, t, y, h):
    """Realiza un paso del método RK4 para la ecuación dydt = f(t, y)."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Ejemplo de uso: resolver una ecuación diferencial
def sistema(t, y):
    return -y + t

# Condiciones iniciales
y0 = 1
t0 = 0
h = 0.1
num_steps = 100

# Tiempo y solución
t_values = np.linspace(t0, t0 + num_steps * h, num_steps)
y_values = np.zeros(num_steps)
y_values[0] = y0

for i in range(1, num_steps):
    y_values[i] = rk4_step(sistema, t_values[i-1], y_values[i-1], h)

# Gráfico de los resultados
import matplotlib.pyplot as plt
plt.plot(t_values, y_values, label="Solución RK4")
plt.xlabel("Tiempo")
plt.ylabel("Estado")
plt.legend()
plt.show()
