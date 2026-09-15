"""
Algoritmo de Colonia de Hormigas (Ant Colony Optimization - ACO)
"""

import numpy as np
import random
import matplotlib.pyplot as plt


class AntColonyOptimizer:
    def __init__(self, distancias, n_hormigas=10, n_iteraciones=100,
                 alpha=1.0, beta=2.0, evaporacion=0.5, Q=100):
        """
        Parámetros:
        -----------
        distancias : matriz NxN con las distancias entre ciudades
        n_hormigas : número de hormigas por iteración
        n_iteraciones : número de iteraciones a ejecutar
        alpha : importancia de la feromona (a mayor valor, más se sigue el rastro)
        beta : importancia de la visibilidad/heurística (1/distancia)
        evaporacion : tasa de evaporación de feromona (0 a 1)
        Q : cantidad de feromona depositada por cada hormiga
        """
        self.distancias = distancias
        self.n_ciudades = len(distancias)
        self.n_hormigas = n_hormigas
        self.n_iteraciones = n_iteraciones
        self.alpha = alpha
        self.beta = beta
        self.evaporacion = evaporacion
        self.Q = Q

        # Matriz de feromonas, inicializada con un valor pequeño uniforme
        self.feromonas = np.ones((self.n_ciudades, self.n_ciudades)) * 0.1

        # Matriz de visibilidad (heurística) = 1 / distancia
        with np.errstate(divide='ignore'):
            self.visibilidad = 1 / self.distancias
        self.visibilidad[self.distancias == 0] = 0

        self.mejor_ruta = None
        self.mejor_distancia = float('inf')
        self.historial_mejores = []

    def _probabilidad_siguiente_ciudad(self, ciudad_actual, no_visitadas):
        """Calcula la probabilidad de moverse a cada ciudad no visitada."""
        feromona = self.feromonas[ciudad_actual, no_visitadas] ** self.alpha
        vista = self.visibilidad[ciudad_actual, no_visitadas] ** self.beta
        atractivo = feromona * vista

        suma = atractivo.sum()
        if suma == 0:
            # Si todo es cero, distribución uniforme
            return np.ones(len(no_visitadas)) / len(no_visitadas)
        return atractivo / suma

    def _construir_ruta(self):
        """Construye una ruta completa para una hormiga."""
        ciudad_inicio = random.randint(0, self.n_ciudades - 1)
        ruta = [ciudad_inicio]
        no_visitadas = list(range(self.n_ciudades))
        no_visitadas.remove(ciudad_inicio)

        while no_visitadas:
            ciudad_actual = ruta[-1]
            probabilidades = self._probabilidad_siguiente_ciudad(ciudad_actual, no_visitadas)
            siguiente = np.random.choice(no_visitadas, p=probabilidades)
            ruta.append(siguiente)
            no_visitadas.remove(siguiente)

        return ruta

    def _distancia_ruta(self, ruta):
        """Calcula la distancia total de una ruta (incluyendo el regreso al inicio)."""
        distancia = 0
        for i in range(len(ruta)):
            ciudad_a = ruta[i]
            ciudad_b = ruta[(i + 1) % len(ruta)]
            distancia += self.distancias[ciudad_a, ciudad_b]
        return distancia

    def _actualizar_feromonas(self, rutas, distancias_rutas):
        """Evapora feromona vieja y deposita feromona nueva según las rutas encontradas."""
        # Evaporación
        self.feromonas *= (1 - self.evaporacion)

        # Depósito de feromona nueva
        for ruta, distancia in zip(rutas, distancias_rutas):
            aporte = self.Q / distancia
            for i in range(len(ruta)):
                ciudad_a = ruta[i]
                ciudad_b = ruta[(i + 1) % len(ruta)]
                self.feromonas[ciudad_a, ciudad_b] += aporte
                self.feromonas[ciudad_b, ciudad_a] += aporte  # simétrico

    def ejecutar(self, verbose=True):
        """Ejecuta el algoritmo completo de ACO."""
        for iteracion in range(self.n_iteraciones):
            rutas = []
            distancias_rutas = []

            # Cada hormiga construye una ruta
            for _ in range(self.n_hormigas):
                ruta = self._construir_ruta()
                distancia = self._distancia_ruta(ruta)
                rutas.append(ruta)
                distancias_rutas.append(distancia)

                if distancia < self.mejor_distancia:
                    self.mejor_distancia = distancia
                    self.mejor_ruta = ruta

            # Actualizar feromonas con base en las rutas de esta iteración
            self._actualizar_feromonas(rutas, distancias_rutas)
            self.historial_mejores.append(self.mejor_distancia)

            if verbose and (iteracion % 10 == 0 or iteracion == self.n_iteraciones - 1):
                print(f"Iteración {iteracion:3d} | Mejor distancia hasta ahora: {self.mejor_distancia:.2f}")

        return self.mejor_ruta, self.mejor_distancia

    def graficar_convergencia(self):
        """Grafica cómo mejora la mejor distancia a lo largo de las iteraciones."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.historial_mejores, color='darkorange')
        plt.title("Convergencia del algoritmo ACO")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor distancia encontrada")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("convergencia_aco.png")
        plt.show()

    def graficar_ruta(self, coordenadas):
        """
        Grafica la mejor ruta encontrada.
        coordenadas: lista de tuplas (x, y) para cada ciudad
        """
        ruta = self.mejor_ruta + [self.mejor_ruta[0]]  # regresar al inicio
        x = [coordenadas[c][0] for c in ruta]
        y = [coordenadas[c][1] for c in ruta]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o-', color='steelblue')
        for i, ciudad in enumerate(self.mejor_ruta):
            plt.annotate(str(ciudad), (coordenadas[ciudad][0], coordenadas[ciudad][1]))
        plt.title(f"Mejor ruta encontrada (distancia = {self.mejor_distancia:.2f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("mejor_ruta_aco.png")
        plt.show()


def generar_distancias_desde_coordenadas(coordenadas):
    """Genera una matriz de distancias euclidianas a partir de coordenadas (x, y)."""
    n = len(coordenadas)
    distancias = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coordenadas[i]
                x2, y2 = coordenadas[j]
                distancias[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distancias


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Ejemplo de uso: 10 ciudades con coordenadas aleatorias
    # ------------------------------------------------------------------
    random.seed(42)
    np.random.seed(42)

    n_ciudades = 10
    coordenadas = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_ciudades)]

    matriz_distancias = generar_distancias_desde_coordenadas(coordenadas)

    aco = AntColonyOptimizer(
        distancias=matriz_distancias,
        n_hormigas=20,
        n_iteraciones=100,
        alpha=1.0,      # importancia de la feromona
        beta=3.0,       # importancia de la distancia (heurística)
        evaporacion=0.5,
        Q=100
    )

    mejor_ruta, mejor_distancia = aco.ejecutar()

    print("\n--- RESULTADO FINAL ---")
    print("Mejor ruta encontrada:", mejor_ruta)
    print(f"Distancia total: {mejor_distancia:.2f}")

    # Gráficas (requieren matplotlib)
    aco.graficar_convergencia()
    aco.graficar_ruta(coordenadas)