#######################################################################
# CEIA - 16Co2024 - Algoritmos Evolutivos - TP2 - Ejercicio 3
# Gustavo J. Rivas (a1620) | Myrna L. Degano (a1618)
#######################################################################
# Algoritmo PSO para minimizar paraboloide elíptico.
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
from tabulate import tabulate
from pyswarm import pso

# Obtener parámetros de ejecución
def get_params():

    num_p = input("Número de partículas (DEFAULT: 20): ").strip()
    num_p = int(num_p) if num_p else 20

    num_i = input("Número de iteraciones (DEFAULT: 10): ").strip()
    num_i = int(num_i) if num_i else 20

    c1 = input("Coeficiente de aceleración - Componente cognitivo (DEFAULT: 2): ").strip()
    c1 = float(c1) if c1 else 2.0

    c2 = input("Coeficiente de aceleración  - Componente social (DEFAULT: 2): ").strip()
    c2 = float(c2) if c2 else 2.0

    w = input("Coeficiente de inercia (DEFAULT: 0.7): ").strip()
    w = float(w) if w else 0.7

    l = input("Límite inferior para las variables x e y (DEFAULT: -100): ").strip()
    l = float(l) if l else -100.0

    u = input("Límite superior para las variables x e y (DEFAULT: +100): ").strip()
    u = float(u) if u else 100.0

    return num_p, num_i, c1, c2, w, l, u

# Obtener inputs
def get_inputs(min=-50, max=50):
    while True:
        try:
            print("\nIngrese los valores de \"a\" y \"b\"")
            print(f"Deben ser valores reales entre {min} y {max} (ambos incluidos)")

            a = float(input("a = ").strip())
            b = float(input("b = ").strip())

            if min <= a <= max and min <= b <= max:
                return a, b
            else:
                print("Los valores ingresados no son válidos.")

        except ValueError:
            print("(!) Entrada no válida.")

# Función objetivo (Paraboloide elíptico)
def f_obj(xy):
    global a
    global b

    x, y = xy

    return (x - a)**2 + (y + b)**2

# Gráfico de la función objetivo 3D
def graph_obj_3d(gX, gY, gZ, s1, s2, redP, redV):

    global a
    global b

    fig = plt.figure(figsize=(s1, s2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(gX, gY, gZ, cmap='viridis')

    # Punto rojo en el mínimo encontrado
    ax.scatter(redP[0], redP[1], redV, color='red', marker='o', s=100)

    ax.set_title(f"Gráfico de la función objetivo\nf(x, y) = (x - {a})^2 + (y + {b})^2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    plt.show()

# Gráfico de la función objetivo 2D
def graph_obj_2d(gX, gY, gZ, s1, s2, redP, redV):
    global a
    global b

    plt.figure(figsize=(s1, s2))
    contour = plt.contourf(gX, gY, gZ, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.scatter([redP[0]], [redP[1]], color='red', zorder=5)
    plt.title(f"Gráfico de la función objetivo\nf(x, y) = (x - {a})^2 + (y + {b})^2")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend([f"Mínimo encontrado\n({redP[0]}, {redP[1]})"])
    plt.grid(True)
    plt.show()


# Gráfico de línea para GBest por iteración
def line_graph(gx, gy, s1, s2, bestV):
    global iterations

    plt.figure(figsize=(s1, s2))

    plt.plot(gx, gy, label='Global Best', color='blue', linestyle='-', linewidth=2, marker='o')
    plt.xticks(range(min(gx), max(gx) + 1))

    for i in range(len(gx)):
        plt.annotate(round(gy[i], 2), (gx[i], gy[i]), fontsize=8, textcoords="offset points", xytext=(0, 10),
                     ha='right')

    # Configurar el eje x para que muestre números enteros
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.xlabel("Iteración #")
    plt.ylabel("GBest")
    plt.title("Gráfico de Global Best por iteración")
    plt.legend(
        loc='upper right',
        title=f"Luego de {iterations} iteraciones \n{bestV}",
        fontsize='small',
        frameon=True,  # Mostrar el marco
        edgecolor='black',  # Color del borde del marco
    )
    # plt.legend(f"Global Best luego de {iterations} iteraciones\n{f_gbest}")
    plt.grid(True)
    plt.show()

class CaptureOutput(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def write(self, data):
        super().write(data)
        try:
            self.records.append(data)
        except ValueError:
            pass

#######################################################################
# Desarrollo del algoritmo
#######################################################################
result = [["Iteración #", "GBest - Valor de x", "GBest - Valor de y", "GBest - Valor de f(x, y)"]]
best_values = []

print("\nINGRESE LOS PARÁMETROS PARA LA EJECUCIÓN DEL ALGORITMO (O <ENTER> PARA TOMAR LOS VALORES POR DEFAULT)\n")

# Dimensiones (x, y)
dim = 2

# Obtener parámetros de ejecución
# Cantidad de partículas, máximo de iteraciones
# Coeficientes de aceleración e incercia (c1, c2, w)
# Límites del espacio (inferior lb, superior ub)
particles, iterations, c1, c2, w, lb, ub = get_params()

# Obtener inputs a y b
a, b = get_inputs()

# Inicializar enjambre de partículas (posiciones y velocidades)
swarm = np.random.uniform(lb, ub, (particles, dim))
velocity = np.zeros((particles, dim))

# Personal Best inicial para cada partícula
pbest = swarm.copy()
f_pbest = [f_obj([swarm[i][0], swarm[i][1]]) for i in range(particles)]

# Global Best inicial
gbest = pbest[np.argmin(f_pbest)]
f_gbest = np.min(f_pbest)

# Búsqueda del óptimo
for i in range(iterations): # Máximo de iteraciones

    for p in range(particles):  # Iteración por partícula

        r1, r2 = np.random.rand(), np.random.rand()  # Aleatorios por cada partícula/iteración

        # Em cada dimensión x, y
        for d in range(dim):
            # Actualizar velocidad de la partícula en cada dimensión
            velocity[p][d] = (w * velocity[p][d] + c1 * r1 * (pbest[p][d] - swarm[p][d]) + c2 * r2 * (gbest[d] - swarm[p][d]))

            # Actualizar posición de la partícula
            # # manteniéndola dentro de los limites del espacio de búsqueda
            swarm[p][d] = np.clip(swarm[p][d] + velocity[p][d], lb, ub)

        # Evaluar la función objetivo para la nueva posición de la partícula
        fitness = f_obj([swarm[p][0], swarm[p][1]])

        # Actualizar el mejor personal
        if fitness < f_pbest[p]:
            f_pbest[p] = fitness
            pbest[p] = swarm[p].copy()

            # Actualizar del mejor global
            if fitness < f_gbest:
                f_gbest = fitness
                gbest = swarm[p].copy()

    # Resultados de la iteración
    result.append([i+1, gbest[0], gbest[1], f_gbest])


# Impresión de resultados
print(tabulate(result, headers="firstrow", tablefmt="grid"))
print(f"\nEl valor óptimo es: {f_gbest} para los valores (x, y): {gbest}")


# Gráficos
X, Y = np.meshgrid(np.linspace(lb, ub, 400), np.linspace(lb, ub, 400))
Z = f_obj([X, Y])

x = np.arange(1, iterations + 1)
y = np.array(result[1:])[:, 3]

# Gráfico de la función objetivo en 3D
graph_obj_3d (X, Y, Z, 10, 5, gbest, f_gbest)

# Gráfico de dispersión para la función objetivo
graph_obj_2d (X, Y, Z, 10, 5, gbest, f_gbest)

# Gráfico de GBest por iteración
line_graph (x, y, 10, 5, f_gbest)


#######################################################################
# Resultados utilizando PYSWARM
#######################################################################

# Redirigir la salida estándar para capturar los mensajes de PSO
original_stdout = sys.stdout
capture_output = CaptureOutput()
sys.stdout = capture_output

# Ejecutar PSO en modo Debug
best_pos, best_val = pso(
    f_obj, # Función objetivo
    [lb, lb], # Límites inferiores
    [ub, ub], # Límites superiores
    swarmsize=particles, # Tamaño del enjambre
    maxiter=iterations, # Número máximo de iteraciones
    debug=True # Modo debug
)

# Restaurar la salida estándar
sys.stdout = original_stdout

# PSO Debug
pso_log_arr = capture_output.records
pso_log_str = ''.join(pso_log_arr)
print(pso_log_str)

print(f"El valor óptimo utilizando pyswarm es: {best_val} para los valores (x, y): {best_pos}")

# Gráfico de la función objetivo en 3D
graph_obj_3d (X, Y, Z, 10, 5, best_pos, best_val)

# Gráfico de dispersión para la función objetivo
graph_obj_2d (X, Y, Z, 10, 5, best_pos, best_val)

# Gráfico de GBest por iteración
pso_log_lines = [s for s in pso_log_arr if s.startswith('Best after iteration')]
for v in pso_log_lines:
    vals = v.split()
    best_values.append(float(vals[-1]))

line_graph (x, best_values, 10, 5, best_val)
