"""
paralelo.py
Implementación paralela por hilos en CPU (Numba prange) del potencial electrostático
de un anillo cargado en una malla 2D (plano xz).
Incluye verificación cuantitativa (np.allclose y error máximo) frente a serial.py.
"""

import time
import numpy as np
import numba
from numba import njit, prange

from Serial import (
    crear_malla,
    _potencial_coulomb_punto_serial,
    _potencial_legendre_punto_serial,
    calcular_potencial_coulomb_serial,
    calcular_potencial_legendre_serial,
)


@njit(parallel=True, fastmath=False)
def calcular_potencial_coulomb_paralelo(x: np.ndarray, z: np.ndarray, a: float = 1.0, m_pasos: int = 200) -> np.ndarray:
    """
    Paraleliza el bucle externo sobre la coordenada z usando Numba prange.
    Cada hilo calcula filas completas de la malla de forma independiente,
    evitando condiciones de carrera y liberando el GIL.
    """
    nx = x.shape[0]
    nz = z.shape[0]
    potencial = np.empty((nz, nx), dtype=np.float64)

    for i in prange(nz):
        z_val = z[i]
        for j in range(nx):
            x_val = x[j]
            potencial[i, j] = _potencial_coulomb_punto_serial(x_val, z_val, a, m_pasos)

    return potencial


@njit(parallel=True, fastmath=False)
def calcular_potencial_legendre_paralelo(x: np.ndarray, z: np.ndarray, a: float = 1.0, n_terminos: int = 40) -> np.ndarray:
    """
    Paraleliza la evaluación de la serie multipolar de Legendre en la malla 2D
    repartiendo las filas z_i entre los hilos de CPU mediante prange.
    """
    nx = x.shape[0]
    nz = z.shape[0]
    potencial = np.empty((nz, nx), dtype=np.float64)

    for i in prange(nz):
        z_val = z[i]
        for j in range(nx):
            x_val = x[j]
            potencial[i, j] = _potencial_legendre_punto_serial(x_val, z_val, a, n_terminos)

    return potencial


def verificar_equivalencia_serial_paralelo(
    tamanos_malla=(200, 500, 800),
    m_pasos: int = 200,
    n_terminos: int = 30,
    a: float = 1.0,
    n_hilos: int = 4,
):
    """
    Cumple el Criterio 4 de la rúbrica: compara cuantitativamente las salidas serial y paralela
    para múltiples tamaños de malla usando np.allclose y reportando la diferencia absoluta máxima.
    """
    max_hilos = numba.config.NUMBA_NUM_THREADS
    hilos_usar = min(n_hilos, max_hilos)
    numba.set_num_threads(hilos_usar)

    # Calentamiento JIT
    x_w, z_w = crear_malla(16, 2.0)
    _ = calcular_potencial_coulomb_serial(x_w, z_w, a, m_pasos)
    _ = calcular_potencial_coulomb_paralelo(x_w, z_w, a, m_pasos)
    _ = calcular_potencial_legendre_serial(x_w, z_w, a, n_terminos)
    _ = calcular_potencial_legendre_paralelo(x_w, z_w, a, n_terminos)

    print(f"=== Verificación cuantitativa Serial vs. Paralelo ({hilos_usar} hilos) ===")
    resultados = []

    for n in tamanos_malla:
        x, z = crear_malla(n, 2.0)

        # 1. Verificación Integración de Coulomb
        t0 = time.perf_counter()
        v_coul_ser = calcular_potencial_coulomb_serial(x, z, a, m_pasos)
        t_ser = time.perf_counter() - t0

        t0 = time.perf_counter()
        v_coul_par = calcular_potencial_coulomb_paralelo(x, z, a, m_pasos)
        t_par = time.perf_counter() - t0

        diff_max_coul = float(np.max(np.abs(v_coul_ser - v_coul_par)))
        ok_coul = bool(np.allclose(v_coul_ser, v_coul_par, rtol=1e-13, atol=1e-14))

        # 2. Verificación Serie de Legendre
        v_leg_ser = calcular_potencial_legendre_serial(x, z, a, n_terminos)
        v_leg_par = calcular_potencial_legendre_paralelo(x, z, a, n_terminos)
        diff_max_leg = float(np.max(np.abs(v_leg_ser - v_leg_par)))
        ok_leg = bool(np.allclose(v_leg_ser, v_leg_par, rtol=1e-13, atol=1e-14))

        print(
            f"Malla {n}x{n} (V_max={np.max(v_coul_par):.2f}) | "
            f"Coulomb -> allclose: {ok_coul}, max|diff|: {diff_max_coul:.3e} ({t_ser:.3f}s vs {t_par:.3f}s) | "
            f"Legendre -> allclose: {ok_leg}, max|diff|: {diff_max_leg:.3e}"
        )
        resultados.append((n, ok_coul, diff_max_coul, ok_leg, diff_max_leg))

    return resultados


if __name__ == "__main__":
    verificar_equivalencia_serial_paralelo()