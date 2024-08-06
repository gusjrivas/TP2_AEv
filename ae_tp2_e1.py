#######################################################################
# CEIA - 16Co2024 - Algoritmos Evolutivos - TP2 - Ejercicio 1
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

    num_i = input("Número de iteraciones (DEFAULT: 80): ").strip()
    num_i = int(num_i) if num_i else 80

    c1 = input("Coeficiente de aceleración - Componente cognitivo (DEFAULT: 2): ").strip()
    c1 = float(c1) if c1 else 2.0

    c2 = input("Coeficiente de aceleración  - Componente social (DEFAULT: 2): ").strip()
    c2 = float(c2) if c2 else 2.0

    w = input("Coeficiente de inercia (DEFAULT: 0.7): ").strip()
    w = float(w) if w else 0.7

    # Límites inferior y suprior de x
    l = 0
    u = 4

    return num_p, num_i, c1, c2, w, l, u


# Función objetivo
def f_obj(x):
    return 2*np.sin(x) - (x**2)/2


#######################################################################
# Desarrollo del algoritmo
#######################################################################
def custom_pso(particles, iterations, c1, c2, w, lb, ub):
    result = [["Iteración #", "GBest - Valor de x", "GBest - Valor de f(x)"]]

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

            # Actualizar velocidad y posición de la partícula dentro de los límites del espacio de búsqueda
            velocity[p] = (w * velocity[p] + c1 * r1 * (pbest[p] - swarm[p]) + c2 * r2 * (gbest - swarm[p]))
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

    return result, f_gbest, gbest

#######################################################################

# Obtener parámetros de ejecución
# Cantidad de partículas, máximo de iteraciones
# Coeficientes de aceleración e incercia (c1, c2, w)
print("\nINGRESE LOS PARÁMETROS PARA LA EJECUCIÓN DEL ALGORITMO (O <ENTER> PARA TOMAR LOS VALORES POR DEFAULT)\n")
particles, iterations, c1, c2, w, lb, ub = get_params()
result, f_gbest, gbest = custom_pso (particles, iterations, c1, c2, w, lb, ub)

# Impresión de resultados
print(tabulate(result, headers="firstrow", tablefmt="grid"))
print(f"\nEl valor óptimo es: {f_gbest} para x= {gbest}")

# Gráfico de la función objetivo
x = np.linspace(lb, ub, 400)
y = 2*np.sin(x) - (x**2)/2

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='y = 2*sin(x) - x^2 / 2', color='blue')

# Añadir el punto verde en el valor máximo
plt.scatter(gbest, f_gbest, color='green', zorder=5, label='Valor máximo')
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
            color='green', zorder=5,
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

# Gráfico de comparación de ejecuciones
num_part = [4, 10, 100, 200 , 400]
# mismos parámetros ingresados - iterations, c1, c2, w, lb, ub

comparative = []
for k in num_part:
    result, f_gbest, gbest = custom_pso (k, iterations, c1, c2, w, lb, ub)
    comparative.append ([k, result, f_gbest, gbest])

# Definir el rango de valores para x
x = np.arange(1, iterations + 1)

y = []
y_min = 100
y_max = 0
for j in comparative:
    jbest = np.array ([sublist[-1] for sublist in j[1][1:]])
    y.append([j[0], jbest, j[2]])
    y_min = np.min(jbest) if np.min(jbest) < y_min else y_min
    y_max = np.max(jbest) if np.max(jbest) > y_max else y_max

plt.figure(figsize=(10, 5))

plt.plot(x, y[0][1], label=f'{y[0][0]} partículas - GBest={y[0][2]}', color='blue')
plt.plot(x, y[1][1], label=f'{y[1][0]} partículas - GBest={y[1][2]}', color='red')
plt.plot(x, y[2][1], label=f'{y[2][0]} partículas - GBest={y[2][2]}', color='green')
plt.plot(x, y[3][1], label=f'{y[3][0]} partículas - GBest={y[3][2]}', color='purple')
plt.plot(x, y[4][1], label=f'{y[4][0]} partículas - GBest={y[4][2]}', color='orange')

plt.title('Comparativa con diferente tamaño de enjambre')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.7)

print (f"{y_min} {y_max}")
plt.yticks(np.arange(y_min, y_max, step=(y_max-y_min)/10))

plt.legend()
plt.show()


