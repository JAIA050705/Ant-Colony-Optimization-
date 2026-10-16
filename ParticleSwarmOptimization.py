"""
Particle Swarm Optimization (PSO) - versión sencilla para aprender
Busca el MÍNIMO de una función moviendo un enjambre de partículas.
"""

import math
import random

# =====================================================================
#  PARÁMETROS PARA JUGAR  (cambia estos valores y vuelve a correr)
# =====================================================================

NUM_PARTICULAS = 40     # Tamaño del enjambre. Más = explora mejor, pero es más lento
NUM_ITERACIONES = 20     # Cuántos "pasos" da el enjambre
DIMENSIONES = 2          # Cuántas variables tiene el problema (x, y, z...)

LIMITE_MIN = -10         # El espacio de búsqueda va de LIMITE_MIN ...
LIMITE_MAX = 10          # ... hasta LIMITE_MAX en cada dimensión

W = 0.7                  # Inercia: alto (0.9) = explora, bajo (0.4) = se afina
C1 = 1.5                 # Jalón hacia el mejor punto PERSONAL (cognitivo) (Pbest)
C2 = 1.5                 # Jalón hacia el mejor punto del GRUPO (social) (Gbest)

FUNCION = "esfera"       # Opciones: "esfera", "rastrigin", "ackley"
SEMILLA = 42             # Un número = resultados repetibles. None = distinto cada vez
MOSTRAR_CADA = 5         # Imprimir el progreso cada cuántas iteraciones

# Si quieres empezar las partículas en posiciones específicas, llena esta lista
# con una posición por partícula (ej. [[5, 5], [-8, 3]]) y se usarán en vez de aleatorias.
POSICIONES_INICIALES = None


# =====================================================================
#  FUNCIONES A MINIMIZAR  (todas tienen su mínimo en [0, 0, ...] = 0)
# =====================================================================

def esfera(x):
    """La más fácil: un solo valle suave."""
    return sum(xi ** 2 for xi in x)


def rastrigin(x):
    """Difícil: muchísimos mínimos locales que engañan al enjambre."""
    return 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ackley(x):
    """Intermedia: zona plana con un hoyo profundo en el centro."""
    n = len(x)
    termino1 = -20 * math.exp(-0.2 * math.sqrt(sum(xi ** 2 for xi in x) / n))
    termino2 = -math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / n)
    return termino1 + termino2 + 20 + math.e


FUNCIONES = {"esfera": esfera, "rastrigin": rastrigin, "ackley": ackley}


# =====================================================================
#  EL ALGORITMO
# =====================================================================

def pso():
    f = FUNCIONES[FUNCION]
    if SEMILLA is not None:
        random.seed(SEMILLA)

    # ---- 1. Inicializar el enjambre ----
    posiciones = []
    velocidades = []
    for i in range(NUM_PARTICULAS):
        if POSICIONES_INICIALES is not None:
            pos = list(POSICIONES_INICIALES[i])
        else:
            pos = [random.uniform(LIMITE_MIN, LIMITE_MAX) for _ in range(DIMENSIONES)]
        vel = [random.uniform(-1, 1) for _ in range(DIMENSIONES)]
        posiciones.append(pos)
        velocidades.append(vel)

    # Cada partícula arranca con su pbest = su posición inicial
    pbest = [pos[:] for pos in posiciones]
    pbest_valor = [f(pos) for pos in posiciones]

    # El gbest es el mejor de todos los pbest
    mejor = pbest_valor.index(min(pbest_valor))
    gbest = pbest[mejor][:]
    gbest_valor = pbest_valor[mejor]

    historial = [gbest_valor]  # Para ver cómo mejora con el tiempo

    # ---- 2. Ciclo principal ----
    for iteracion in range(1, NUM_ITERACIONES + 1):
        for i in range(NUM_PARTICULAS):
            for d in range(DIMENSIONES):
                r1 = random.random()
                r2 = random.random()

                # Actualizar velocidad: inercia + jalón personal + jalón del grupo
                inercia = W * velocidades[i][d]
                cognitivo = C1 * r1 * (pbest[i][d] - posiciones[i][d])
                social = C2 * r2 * (gbest[d] - posiciones[i][d])
                velocidades[i][d] = inercia + cognitivo + social

                # Actualizar posición
                posiciones[i][d] += velocidades[i][d]

                # Mantener la partícula dentro de los límites
                posiciones[i][d] = max(LIMITE_MIN, min(LIMITE_MAX, posiciones[i][d]))

            # Evaluar y actualizar récords
            valor = f(posiciones[i])
            if valor < pbest_valor[i]:
                pbest[i] = posiciones[i][:]
                pbest_valor[i] = valor
                if valor < gbest_valor:
                    gbest = posiciones[i][:]
                    gbest_valor = valor

        historial.append(gbest_valor)

        if iteracion % MOSTRAR_CADA == 0 or iteracion == 1:
            pos_txt = ", ".join(f"{v:8.4f}" for v in gbest)
            print(f"Iteración {iteracion:3d} | mejor valor = {gbest_valor:12.6f} | posición = [{pos_txt}]")

    return gbest, gbest_valor, historial


# =====================================================================
#  EJECUTAR
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f" PSO | función: {FUNCION} | partículas: {NUM_PARTICULAS} | "
          f"w={W}  c1={C1}  c2={C2}")
    print("=" * 70)

    mejor_pos, mejor_valor, historial = pso()

    print("=" * 70)
    print("RESULTADO FINAL")
    print(f"  Mejor posición : {[round(v, 5) for v in mejor_pos]}")
    print(f"  Mejor valor    : {mejor_valor:.8f}   (el ideal es 0)")
    print("=" * 70)

    # Gráfica opcional de la convergencia (solo si tienes matplotlib instalado)
    try:
        import matplotlib.pyplot as plt

        plt.plot(historial, linewidth=2)
        plt.title(f"Convergencia de PSO - función {FUNCION}")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor valor encontrado (gbest)")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("(Instala matplotlib con 'pip install matplotlib' para ver la gráfica)")