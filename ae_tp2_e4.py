#######################################################################
# CEIA - 16Co2024 - Algoritmos Evolutivos - TP2 - Ejercicio 4
# Gustavo J. Rivas (a1620) | Myrna L. Degano (a1618)
#######################################################################
# Sistema de 2 ecuaciones con 2 incógnitas resuelto con PSO.
#######################################################################

from pyswarm import pso

# Obtener parámetros de ejecución
def get_params():

    num_p = input("Número de partículas (DEFAULT: 20): ").strip()
    num_p = int(num_p) if num_p else 20

    num_i = input("Número de iteraciones (DEFAULT: 50): ").strip()
    num_i = int(num_i) if num_i else 50

    c1 = input("Coeficiente de aceleración - Componente cognitivo (DEFAULT: 1.5): ").strip()
    c1 = float(c1) if c1 else 1.5

    c2 = input("Coeficiente de aceleración  - Componente social (DEFAULT: 1.5): ").strip()
    c2 = float(c2) if c2 else 1.5

    w = input("Coeficiente de inercia (DEFAULT: 0.5): ").strip()
    w = float(w) if w else 0.5

    l = input("Límite inferior para las variables x e y (DEFAULT: -100): ").strip()
    l = float(l) if l else -100.0

    u = input("Límite superior para las variables x e y (DEFAULT: +100): ").strip()
    u = float(u) if u else 100.0

    return num_p, num_i, c1, c2, w, l, u

# Función objetivo (Suma de cuadrados de las ecuaciones -> Error a minimizar)
def f_obj(x1x2):

    x1, x2 = x1x2

    # Sistema de ecuaciones
    # 3*x1 + 2*x2 - 9 = 0 => f1 (x1, x2) = 3*x1 + 2*x2 - 9
    # x1 - 5*x2 - 4 = 0 => f1 (x1, x2) = x1 - 5*x2 - 4

    # Función Objetivo a minimizar: f1^2 + f2^2
    return (3*x1 + 2*x2 - 9)**2 + (x1 - 5*x2 - 4)**2


#######################################################################
# Desarrollo del algoritmo
#######################################################################

print("\nINGRESE LOS PARÁMETROS PARA LA EJECUCIÓN DEL ALGORITMO (O <ENTER> PARA TOMAR LOS VALORES POR DEFAULT)\n")

# Obtener parámetros de ejecución
# Cantidad de partículas, máximo de iteraciones
# Coeficientes de aceleración e incercia (c1, c2, w)
# Límites del espacio (inferior lb, superior ub)
particles, iterations, c1, c2, w, lb, ub = get_params()

# Ejecutar PSO en modo Debug
best_pos, best_val = pso(
    f_obj, # Función objetivo
    [lb, lb], # Límites inferiores
    [ub, ub], # Límites superiores
    swarmsize=particles, # Tamaño del enjambre
    maxiter=iterations, # Número máximo de iteraciones
    debug=True # Modo debug
)

print(f"\nLos valores aproximados encontrados para resolver el sistema de ecuaciones son:\n x1, x2 = {best_pos}")

