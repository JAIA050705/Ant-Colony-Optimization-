"""
Algoritmo de Selección Clonal (CLONALG) - Sistemas Inmunológicos Artificiales
Busca el MÍNIMO de una función clonando y mutando las mejores soluciones.

Idea: los "anticuerpos" (soluciones) que mejor encajan se clonan mucho,
y los clones mutan: los buenos mutan poco (afinan), los malos mutan mucho (exploran).
"""

import math
import random

# =====================================================================
#  PARÁMETROS PARA JUGAR  (cambia estos valores y vuelve a correr)
# =====================================================================

NUM_ANTICUERPOS = 20     # Tamaño de la población (las soluciones candidatas)
NUM_ITERACIONES = 50     # Cuántas generaciones se repite el proceso
DIMENSIONES = 2          # Cuántas variables tiene el problema (x, y, z...)

LIMITE_MIN = -10         # El espacio de búsqueda va de LIMITE_MIN ...
LIMITE_MAX = 10          # ... hasta LIMITE_MAX en cada dimensión

N_MEJORES = 10           # Cuántos anticuerpos se seleccionan para clonar cada ronda
FACTOR_CLONACION = 1.0   # Más alto = más clones. El mejor recibe N_MEJORES*factor clones,
                         # el segundo la mitad, el tercero un tercio, etc.
RHO = 2.0                # Qué tanto se protege a los mejores de la mutación.
                         # Alto = los buenos casi no mutan. Bajo = todos mutan parecido
ESCALA_MUTACION = 0.3    # Tamaño base de la mutación (fracción del rango de búsqueda)
DECAIMIENTO = 0.90       # Cada ronda la mutación se multiplica por esto (se afina con el tiempo).
                         # Pon 1.0 para que no se reduzca

NUM_REEMPLAZO = 3        # Cuántos de los PEORES anticuerpos se cambian por otros nuevos
                         # aleatorios en cada ronda (mantiene la diversidad)

FUNCION = "ackley"       # Opciones: "esfera", "rastrigin", "ackley"
SEMILLA = 42             # Un número = resultados repetibles. None = distinto cada vez
MOSTRAR_CADA = 5         # Imprimir el progreso cada cuántas iteraciones

# Si quieres empezar los anticuerpos en posiciones específicas, llena esta lista
# con una posición por anticuerpo (ej. [[5, 5], [-8, 3]]) y se usarán en vez de aleatorias.
POSICIONES_INICIALES = None


# =====================================================================
#  FUNCIONES A MINIMIZAR  (todas tienen su mínimo en [0, 0, ...] = 0)
# =====================================================================

def esfera(x):
    """La más fácil: un solo valle suave."""
    return sum(xi ** 2 for xi in x)


def rastrigin(x):
    """Difícil: muchísimos mínimos locales que engañan al algoritmo."""
    return 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ackley(x):
    """Intermedia: zona plana con un hoyo profundo en el centro."""
    n = len(x)
    termino1 = -20 * math.exp(-0.2 * math.sqrt(sum(xi ** 2 for xi in x) / n))
    termino2 = -math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / n)
    return termino1 + termino2 + 20 + math.e


FUNCIONES = {"esfera": esfera, "rastrigin": rastrigin, "ackley": ackley}


# =====================================================================
#  FUNCIONES AUXILIARES
# =====================================================================

def posicion_aleatoria():
    """Crea un anticuerpo nuevo en un lugar al azar del espacio de búsqueda."""
    return [random.uniform(LIMITE_MIN, LIMITE_MAX) for _ in range(DIMENSIONES)]


def calcular_afinidades(valores):
    """
    Convierte los valores de la función en una afinidad entre 0 y 1.
    Como buscamos el MÍNIMO: el mejor anticuerpo tiene afinidad 1 y el peor 0.
    """
    vmin, vmax = min(valores), max(valores)
    if vmax == vmin:
        return [1.0] * len(valores)
    return [1 - (v - vmin) / (vmax - vmin) for v in valores]


def mutar(anticuerpo, afinidad, escala):
    """
    Hipermutación: cuanto MAYOR la afinidad, MENOR la mutación.
    Los buenos solo se afinan; los malos exploran lejos.
    """
    rango = LIMITE_MAX - LIMITE_MIN
    sigma = escala * rango * math.exp(-RHO * afinidad)

    clon = []
    for xi in anticuerpo:
        nuevo = xi + random.gauss(0, sigma)
        nuevo = max(LIMITE_MIN, min(LIMITE_MAX, nuevo))  # mantenerlo dentro de los límites
        clon.append(nuevo)
    return clon


# =====================================================================
#  EL ALGORITMO
# =====================================================================

def clonalg():
    f = FUNCIONES[FUNCION]
    if SEMILLA is not None:
        random.seed(SEMILLA)

    evaluaciones = 0  # Cuántas veces se calculó la función (para comparar con PSO)

    # ---- 1. Crear la población inicial de anticuerpos ----
    if POSICIONES_INICIALES is not None:
        poblacion = [list(p) for p in POSICIONES_INICIALES]
    else:
        poblacion = [posicion_aleatoria() for _ in range(NUM_ANTICUERPOS)]

    valores = [f(a) for a in poblacion]
    evaluaciones += len(poblacion)

    mejor_valor = min(valores)
    mejor = poblacion[valores.index(mejor_valor)][:]
    historial = [mejor_valor]

    escala = ESCALA_MUTACION

    # ---- 2. Ciclo principal ----
    for iteracion in range(1, NUM_ITERACIONES + 1):

        # Afinidad de cada anticuerpo (1 = el mejor, 0 = el peor)
        afinidades = calcular_afinidades(valores)

        # Ordenar de mejor a peor y quedarse con los N mejores
        orden = sorted(range(len(poblacion)), key=lambda i: valores[i])
        seleccionados = orden[:N_MEJORES]

        # ---- 3. Clonar y mutar ----
        for lugar, i in enumerate(seleccionados, start=1):
            # El mejor (lugar 1) recibe más clones; el siguiente la mitad, etc.
            num_clones = max(1, round(FACTOR_CLONACION * N_MEJORES / lugar))

            mejor_clon = None
            mejor_clon_valor = float("inf")

            for _ in range(num_clones):
                clon = mutar(poblacion[i], afinidades[i], escala)
                valor_clon = f(clon)
                evaluaciones += 1
                if valor_clon < mejor_clon_valor:
                    mejor_clon, mejor_clon_valor = clon, valor_clon

            # Si el mejor clon supera al original, lo reemplaza
            if mejor_clon_valor < valores[i]:
                poblacion[i] = mejor_clon
                valores[i] = mejor_clon_valor

        # ---- 4. Reemplazar a los peores por anticuerpos nuevos (diversidad) ----
        peores = sorted(range(len(poblacion)), key=lambda i: valores[i], reverse=True)
        for i in peores[:NUM_REEMPLAZO]:
            poblacion[i] = posicion_aleatoria()
            valores[i] = f(poblacion[i])
            evaluaciones += 1

        # ---- 5. Actualizar el mejor global ----
        for i in range(len(poblacion)):
            if valores[i] < mejor_valor:
                mejor_valor = valores[i]
                mejor = poblacion[i][:]

        historial.append(mejor_valor)
        escala *= DECAIMIENTO  # la mutación se vuelve más fina con el tiempo

        if iteracion % MOSTRAR_CADA == 0 or iteracion == 1:
            pos_txt = ", ".join(f"{v:8.4f}" for v in mejor)
            print(f"Iteración {iteracion:3d} | mejor valor = {mejor_valor:12.6f} | posición = [{pos_txt}]")

    return mejor, mejor_valor, historial, evaluaciones


# =====================================================================
#  EJECUTAR
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f" CLONALG | función: {FUNCION} | anticuerpos: {NUM_ANTICUERPOS} | "
          f"mejores: {N_MEJORES}  rho={RHO}")
    print("=" * 70)

    mejor_pos, mejor_valor, historial, evaluaciones = clonalg()

    print("=" * 70)
    print("RESULTADO FINAL")
    print(f"  Mejor posición         : {[round(v, 5) for v in mejor_pos]}")
    print(f"  Mejor valor            : {mejor_valor:.8f}   (el ideal es 0)")
    print(f"  Evaluaciones de la f(x): {evaluaciones}")
    print("=" * 70)

    # Gráfica opcional de la convergencia (solo si tienes matplotlib instalado)
    try:
        import matplotlib.pyplot as plt

        plt.plot(historial, linewidth=2)
        plt.title(f"Convergencia de CLONALG - función {FUNCION}")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor valor encontrado")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("(Instala matplotlib con 'pip install matplotlib' para ver la gráfica)")