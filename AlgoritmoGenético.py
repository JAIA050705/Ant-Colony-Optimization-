"""
Algoritmo Genético para optimizar una FUNCIÓN OBJETIVO real.
Ejemplo: maximizar f(x) = x * sin(10*pi*x) + 1   en el rango [0, 1]

Para usar tu propia función, solo cambia:
  - la función `funcion_objetivo(x)`
  - el rango [LIMITE_INF, LIMITE_SUP]
"""

import random
import math

# ---------- Parámetros ----------
TAM_CROMOSOMA = 20        # bits usados para representar el número real
TAM_POBLACION = 50
PROB_CRUCE = 0.8
PROB_MUTACION = 0.02
GENERACIONES = 100
TAM_TORNEO = 3

LIMITE_INF = 0.0
LIMITE_SUP = 1.0


# ---------- Función objetivo (¡AQUÍ VA TU PROBLEMA REAL!) ----------

def funcion_objetivo(x):
    """Función a maximizar. Cambia esto por tu propia función."""
    return x * math.sin(10 * math.pi * x) + 1


# ---------- Decodificación binaria -> número real ----------

def decodificar(cromosoma):
    """Convierte una cadena de bits en un número real dentro del rango."""
    entero = int("".join(map(str, cromosoma)), 2)
    max_entero = 2 ** TAM_CROMOSOMA - 1
    return LIMITE_INF + (entero / max_entero) * (LIMITE_SUP - LIMITE_INF)


def fitness(individuo):
    x = decodificar(individuo)
    return funcion_objetivo(x)


# ---------- Operadores genéticos ----------

def crear_individuo():
    return [random.randint(0, 1) for _ in range(TAM_CROMOSOMA)]


def crear_poblacion():
    return [crear_individuo() for _ in range(TAM_POBLACION)]


def seleccion_torneo(poblacion):
    competidores = random.sample(poblacion, TAM_TORNEO)
    return max(competidores, key=fitness)


def cruce_un_punto(padre1, padre2):
    if random.random() < PROB_CRUCE:
        punto = random.randint(1, TAM_CROMOSOMA - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1[:], padre2[:]


def mutar(individuo):
    return [
        (1 - bit) if random.random() < PROB_MUTACION else bit
        for bit in individuo
    ]


# ---------- Bucle principal ----------

def algoritmo_genetico():
    poblacion = crear_poblacion()
    mejor_historico = max(poblacion, key=fitness)

    for generacion in range(GENERACIONES):
        nueva_poblacion = []

        # Elitismo
        mejor_actual = max(poblacion, key=fitness)
        nueva_poblacion.append(mejor_actual[:])

        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)

            hijo1, hijo2 = cruce_un_punto(padre1, padre2)
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < TAM_POBLACION:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

        mejor_actual = max(poblacion, key=fitness)
        if fitness(mejor_actual) > fitness(mejor_historico):
            mejor_historico = mejor_actual

        if generacion % 10 == 0:
            x = decodificar(mejor_actual)
            print(f"Generación {generacion:3d} | "
                  f"x = {x:.5f} | f(x) = {fitness(mejor_actual):.5f}")

    x_final = decodificar(mejor_historico)
    print("\n--- Resultado final ---")
    print(f"Mejor x encontrado: {x_final:.6f}")
    print(f"f(x) máximo: {fitness(mejor_historico):.6f}")


if __name__ == "__main__":
    algoritmo_genetico()