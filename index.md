# Bienvenido al módulo RK4

Esta documentación describe el uso del módulo RK4, una implementación del método Runge-Kutta de cuarto orden, utilizada para resolver sistemas dinámicos. El código fuente del módulo está disponible en: [GitHub](https://github.com).

## Sistemas Dinámicos

Los sistemas dinámicos son modelos fundamentales en ciencias como la física, donde el estado de un sistema cambia con el tiempo. Estos sistemas se describen mediante partículas o conjuntos de partículas cuyo comportamiento está gobernado por ecuaciones diferenciales que involucran derivadas temporales.

En general, un modelo dinámico busca predecir cómo evoluciona una cantidad física a lo largo del tiempo, lo cual se hace mediante el uso de generadores dinámicos, a menudo representados de forma funcional.

### Ecuación Dinámica

En muchos casos, podemos describir la dinámica de un sistema mediante una ecuación diferencial de la forma:

$$
\frac{dy}{dt} = f(t, y),
$$

con una condición inicial dada por:

$$
y(t_0) = y_0.
$$

Donde:
- \( t \) representa la variable temporal,
- \( y \) es el estado del sistema, que puede ser representado por distintos objetos matemáticos, desde un escalar hasta una matriz que describe un operador lineal.

Este tipo de problema es conocido en matemáticas aplicadas como un problema de valor inicial, ya que se busca determinar la evolución de un sistema desde un estado conocido en un tiempo inicial.

## Implementación del Método RK4

El método de Runge-Kutta de cuarto orden (RK4) es un esquema numérico muy utilizado para resolver problemas de ecuaciones diferenciales como el descrito anteriormente. A continuación se presenta un ejemplo de cómo utilizar el módulo RK4 para resolver un sistema dinámico sencillo.

```python
import numpy as np

def rk4_step(f, t, y, h):
    """Realiza un paso del método RK4 para la ecuación dydt = f(t, y)."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Ejemplo de uso: resolver una ecuación diferencial sencilla
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
