#######################################################################
# CEIA - 16Co2024 - Algoritmos Evolutivos - TP2 - Ejercicio 2
# Gustavo J. Rivas (a1620) | Myrna L. Degano (a1618)
#######################################################################
# Algoritmo PSO para maximizar función.
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Obtener parámetros de ejecución
def get_params():

    num_p = input("Número de partículas (DEFAULT: 2): ").strip()
    num_p = int(num_p) if num_p else 2

    num_i = input("Número de iteraciones (DEFAULT: 30): ").strip()
    num_i = int(num_i) if num_i else 30

    c1 = input("Coeficiente de aceleración - Componente cognitivo (DEFAULT: 1.49): ").strip()
    c1 = float(c1) if c1 else 1.49

    c2 = input("Coeficiente de aceleración  - Componente social (DEFAULT: 1.49): ").strip()
    c2 = float(c2) if c2 else 1.49

    w = input("Coeficiente de inercia (DEFAULT: 0.5): ").strip()
    w = float(w) if w else 0.5

    l = 0
    u = 10

    return num_p, num_i, c1, c2, w, l, u

# Función objetivo
def f_obj(x):
    return np.sin(x) + np.sin(x**2)


#######################################################################
# Desarrollo del algoritmo
#######################################################################
result = [["Iteración #", "GBest - Valor de x", "GBest - Valor de f(x)"]]

print("\nINGRESE LOS PARÁMETROS PARA LA EJECUCIÓN DEL ALGORITMO (O <ENTER> PARA TOMAR LOS VALORES POR DEFAULT)\n")

# Obtener parámetros de ejecución
# Cantidad de partículas, máximo de iteraciones
# Coeficientes de aceleración e incercia (c1, c2, w)
# Límites del espacio (inferior lb, superior ub)
particles, iterations, c1, c2, w, lb, ub = get_params()

# Inicializar enjambre de partículas (posiciones y velocidades)
swarm = np.random.uniform(lb, ub, particles)
velocity = np.zeros(particles)

# Personal Best inicial para cada partícula
pbest = swarm.copy()
f_pbest = [f_obj(swarm[i]) for i in range(particles)]

# Global Best inicial
gbest = pbest[np.argmax(f_pbest)]
f_gbest = np.max(f_pbest)

# Búsqueda del óptimo
for i in range(iterations): # Máximo de iteraciones

    for p in range(particles):  # Iteración por partícula

        r1, r2 = np.random.rand(), np.random.rand()  # Aleatorios por cada partícula/iteración

        # Actualizar velocidad de la partícula
        velocity[p] = (w * velocity[p] + c1 * r1 * (pbest[p] - swarm[p]) + c2 * r2 * (gbest - swarm[p]))

        # Actualizar posición de la partícula
        # # manteniéndola dentro de los limites del espacio de búsqueda
        swarm[p] = np.clip(swarm[p] + velocity[p], lb, ub)

        # Evaluar la función objetivo para la nueva posición de la partícula
        fitness = f_obj(swarm[p])

        # Actualizar el mejor personal
        if fitness > f_pbest[p]:
            f_pbest[p] = fitness
            pbest[p] = swarm[p].copy()

            # Actualizar del mejor global
            if fitness > f_gbest:
                f_gbest = fitness
                gbest = swarm[p].copy()

    # Resultados de la iteración
    result.append([i+1, gbest, f_gbest])


# Impresión de resultados
print(tabulate(result, headers="firstrow", tablefmt="grid"))
print(f"\nEl valor óptimo es: {f_gbest} para x= {gbest}")

# Gráfico de la función objetivo
x = np.linspace(lb, ub, 400)
y = np.sin(x) + np.sin(x**2)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='y = sin(x) + sin(x^2)', color='blue')

# Añadir el punto negro en el valor máximo
plt.scatter(gbest, f_gbest, color='black', zorder=5, label='Valor máximo')
plt.annotate(f'GBest\nx={gbest}\nf(x)={f_gbest}', (gbest, f_gbest), textcoords="offset points", xytext=(0,-40), ha='center', zorder=10)

plt.title('Gráfica de la función objetivo')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Gráfico de GBest en función de las iteraciones
x = np.arange(1, iterations + 1)
y = np.array(result[1:])[:, 2]

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='GBest en cada iteración', color='blue', linewidth=2)
plt.title("Gráfica de Global Best en cada Iteración")
plt.xlabel('Iteración #')
plt.ylabel('Global Best')
plt.scatter(iterations, f_gbest,
            color='black', zorder=5,
            label=f'Valor máximo luego de {iterations} iteraciones = {f_gbest}')

step = iterations // 10 if iterations > 30 else 1
for i in range(0, len(x), step):
    plt.annotate(round(y[i], 2),
                 (x[i], y[i]),
                 fontsize=8,
                 textcoords="offset points",
                 xytext=(0, -15),
                 ha='right',
                 arrowprops=dict(facecolor='blue', shrink=0, width=0.5, headwidth=5, headlength=5))

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
