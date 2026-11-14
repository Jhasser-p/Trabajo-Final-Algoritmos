"""
medir.py
Script maestro de evaluación de desempeño, verificación y generación de figuras.
Cumple con los Criterios 4, 5, 6 y 7 de la rúbrica de Cálculo Científico:
1. Reporta las especificaciones de hardware y versiones de software.
2. Mide y reporta por separado el tiempo de compilación JIT (calentamiento).
3. Ejecuta experimentos de tiempo con time.perf_counter (>= 5 repeticiones),
   calculando mediana, desviación estándar y rango intercuartílico (IQR).
4. Varía el número de hilos (1, 2, 4, ... hasta el máximo de la máquina) en dos tamaños
   de malla distintos y exporta todos los datos a 'resultados_desempeno.csv'.
5. Estima la fracción paralelizable (f) mediante ajuste por mínimos cuadrados a la Ley de Amdahl.
6. Evalúa el colapso de convergencia multipolar cuando r -> a y exporta 'colapso_convergencia.csv'.
7. Regenera automáticamente todas las figuras del informe y del póster.
"""

import csv
import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import numba

from Serial import (
    crear_malla,
    _potencial_coulomb_punto_serial,
    _potencial_legendre_punto_serial,
    calcular_potencial_coulomb_serial,
    calcular_potencial_legendre_serial,
    graficar_fenomeno_y_error,
)
from Paralelo import (
    calcular_potencial_coulomb_paralelo,
    calcular_potencial_legendre_paralelo,
    verificar_equivalencia_serial_paralelo,
)


def obtener_lista_hilos(max_hilos: int) -> list:
    """
    Construye la secuencia de hilos a evaluar: 1, 2, 4, 8, ... hasta max_hilos.
    Garantiza que max_hilos esté incluido en la lista.
    """
    hilos = []
    p = 1
    while p < max_hilos:
        hilos.append(p)
        p *= 2
    if max_hilos not in hilos:
        hilos.append(max_hilos)
    # Si la máquina tiene más de 4 hilos, incluimos un punto intermedio adicional (p. ej. 6 o 12)
    if max_hilos >= 6 and (max_hilos // 2 + 2) < max_hilos:
        intermedio = max_hilos - 2
        if intermedio not in hilos and intermedio > 2:
            hilos.append(intermedio)
    return sorted(hilos)


def reportar_hardware() -> dict:
    """
    Recopila e imprime la información del hardware y entorno de software.
    """
    try:
        capa_hilos = numba.threading_layer()
    except (ValueError, AttributeError):
        capa_hilos = str(getattr(numba.config, "THREADING_LAYER", "default"))

    info = {
        "sistema": f"{platform.system()} {platform.release()}",
        "arquitectura": platform.machine(),
        "procesador": platform.processor() or "CPU x86_64 / ARM64",
        "nucleos_logicos_os": os.cpu_count() or 1,
        "hilos_numba_max": numba.config.NUMBA_NUM_THREADS,
        "capa_hilos_numba": capa_hilos,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
    }

    print("=" * 72)
    print("ESPECIFICACIONES DEL ENTORNO Y HARDWARE")
    print("=" * 72)
    print(f"  Sistema Operativo : {info['sistema']} ({info['arquitectura']})")
    print(f"  Procesador        : {info['procesador']}")
    print(f"  Núcleos detectados: {info['nucleos_logicos_os']} lógicos (Numba máx: {info['hilos_numba_max']}, capa: {info['capa_hilos_numba']})")
    print(f"  Versiones         : Python {info['python_version']} | NumPy {info['numpy_version']} | Numba {info['numba_version']}")
    print("=" * 72)
    return info


def medir_compilacion_jit(a: float = 1.0, m_pasos: int = 200, k_terminos: int = 40) -> dict:
    """
    Mide por separado el tiempo de la primera llamada (que incluye la compilación JIT de LLVM)
    usando una malla pequeña de 16x16 puntos. Así se excluye la compilación de las mediciones
    posteriores y se reporta de forma transparente (Criterio 5).
    """
    x_warm, z_warm = crear_malla(16, 2.0)

    t0 = time.perf_counter()
    _ = calcular_potencial_coulomb_serial(x_warm, z_warm, a, m_pasos)
    t_jit_coul_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = calcular_potencial_coulomb_paralelo(x_warm, z_warm, a, m_pasos)
    t_jit_coul_par = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = calcular_potencial_legendre_serial(x_warm, z_warm, a, k_terminos)
    t_jit_leg_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = calcular_potencial_legendre_paralelo(x_warm, z_warm, a, k_terminos)
    t_jit_leg_par = time.perf_counter() - t0

    tiempos_jit = {
        "coulomb_serial_jit_s": t_jit_coul_ser,
        "coulomb_paralelo_jit_s": t_jit_coul_par,
        "legendre_serial_jit_s": t_jit_leg_ser,
        "legendre_paralelo_jit_s": t_jit_leg_par,
    }

    print("\n=== Tiempos de compilación JIT en frío (excluidos del benchmark) ===")
    print(f"  Coulomb Serial   : {t_jit_coul_ser:.4f} s | Coulomb Paralelo  : {t_jit_coul_par:.4f} s")
    print(f"  Legendre Serial  : {t_jit_leg_ser:.4f} s | Legendre Paralelo : {t_jit_leg_par:.4f} s\n")
    return tiempos_jit


def estimar_fraccion_amdahl(hilos: np.ndarray, speedups: np.ndarray) -> float:
    """
    Estima la fracción paralelizable 'f' de la Ley de Amdahl:
        S_p = 1 / ((1 - f) + f / p)  =>  1 - 1 / S_p = f * (1 - 1 / p)
    Resuelve por mínimos cuadrados lineales para p > 1.
    """
    mascara = hilos > 1
    if not np.any(mascara):
        return 0.95

    x = 1.0 - (1.0 / hilos[mascara].astype(np.float64))
    y = 1.0 - (1.0 / speedups[mascara].astype(np.float64))

    f_est = float(np.sum(x * y) / np.sum(x * x))
    return float(np.clip(f_est, 0.0, 0.9999))


def ejecutar_benchmark_desempeno(
    tamanos_malla=(600, 1500),
    repeticiones: int = 5,
    m_pasos: int = 200,
    k_terminos: int = 60,
    a: float = 1.0,
    ruta_csv: str = "resultados_desempeno.csv",
):
    """
    Ejecuta las mediciones de tiempo serial y paralelo para al menos dos tamaños de problema
    y múltiples conteos de hilos, repitiendo cada configuración >= 5 veces.
    Guarda los resultados detallados en un archivo CSV.
    """
    max_hilos = numba.config.NUMBA_NUM_THREADS
    lista_hilos = obtener_lista_hilos(max_hilos)

    algoritmos = [
        (
            "Coulomb_Simpson",
            m_pasos,
            calcular_potencial_coulomb_serial,
            calcular_potencial_coulomb_paralelo,
        ),
        (
            "Legendre_Serie",
            k_terminos,
            calcular_potencial_legendre_serial,
            calcular_potencial_legendre_paralelo,
        ),
    ]

    registros = []
    print(f"=== Iniciando Benchmark de Desempeño ({repeticiones} repeticiones por punto) ===")
    print(f"  Tamaños de malla : {tamanos_malla}")
    print(f"  Hilos evaluados  : {lista_hilos}\n")

    for nombre_alg, param_interno, func_serial, func_paralelo in algoritmos:
        for n in tamanos_malla:
            x, z = crear_malla(n, 2.0)

            # 1. Medición de la versión serial base (T_1)
            tiempos_ser = []
            for _ in range(repeticiones):
                t0 = time.perf_counter()
                _ = func_serial(x, z, a, param_interno)
                tiempos_ser.append(time.perf_counter() - t0)

            t1_mediana = float(np.median(tiempos_ser))
            t1_std = float(np.std(tiempos_ser, ddof=1)) if repeticiones > 1 else 0.0
            t1_iqr = float(np.percentile(tiempos_ser, 75) - np.percentile(tiempos_ser, 25))

            registros.append({
                "algoritmo": nombre_alg,
                "parametro_interno": param_interno,
                "tamano_malla_N": n,
                "puntos_totales": n * n,
                "version": "serial",
                "hilos": 1,
                "repeticiones": repeticiones,
                "tiempo_mediana_s": t1_mediana,
                "tiempo_promedio_s": float(np.mean(tiempos_ser)),
                "tiempo_std_s": t1_std,
                "tiempo_iqr_s": t1_iqr,
                "tiempo_min_s": float(np.min(tiempos_ser)),
                "tiempo_max_s": float(np.max(tiempos_ser)),
                "speedup": 1.0,
                "eficiencia": 1.0,
            })

            print(
                f"[{nombre_alg} | N={n}x{n}] Serial (1 hilo) -> "
                f"Mediana: {t1_mediana:.4f} s (±{t1_std:.4f} s, IQR: {t1_iqr:.4f} s)"
            )

            # 2. Medición de la versión paralela variando el número de hilos p
            for p in lista_hilos:
                numba.set_num_threads(p)
                tiempos_par = []
                for _ in range(repeticiones):
                    t0 = time.perf_counter()
                    _ = func_paralelo(x, z, a, param_interno)
                    tiempos_par.append(time.perf_counter() - t0)

                tp_mediana = float(np.median(tiempos_par))
                tp_std = float(np.std(tiempos_par, ddof=1)) if repeticiones > 1 else 0.0
                tp_iqr = float(np.percentile(tiempos_par, 75) - np.percentile(tiempos_par, 25))

                speedup = t1_mediana / tp_mediana
                eficiencia = speedup / p

                registros.append({
                    "algoritmo": nombre_alg,
                    "parametro_interno": param_interno,
                    "tamano_malla_N": n,
                    "puntos_totales": n * n,
                    "version": "paralelo",
                    "hilos": p,
                    "repeticiones": repeticiones,
                    "tiempo_mediana_s": tp_mediana,
                    "tiempo_promedio_s": float(np.mean(tiempos_par)),
                    "tiempo_std_s": tp_std,
                    "tiempo_iqr_s": tp_iqr,
                    "tiempo_min_s": float(np.min(tiempos_par)),
                    "tiempo_max_s": float(np.max(tiempos_par)),
                    "speedup": speedup,
                    "eficiencia": eficiencia,
                })

                print(
                    f"[{nombre_alg} | N={n}x{n}] Paralelo ({p:2d} hilos) -> "
                    f"Mediana: {tp_mediana:.4f} s (±{tp_std:.4f} s) | "
                    f"Speedup: {speedup:.2f}x | Eficiencia: {eficiencia * 100:.1f}%"
                )
            print("-" * 72)

    # Exportar a CSV
    campos = list(registros[0].keys())
    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)

    print(f"Datos de desempeño guardados en '{ruta_csv}'.\n")
    return registros


def graficar_resultados_desempeno(
    registros: list,
    ruta_salida: str = "desempeno_speedup_eficiencia.png",
):
    """
    Genera las gráficas de Tiempo de Ejecución, Speedup (con línea ideal y curva de Amdahl)
    y Eficiencia vs. Número de Hilos para todos los algoritmos y tamaños de malla.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    # Agrupar series paralelas por (algoritmo, tamano_malla_N)
    claves = sorted(list({(r["algoritmo"], r["tamano_malla_N"]) for r in registros if r["version"] == "paralelo"}))
    marcadores = ["o", "s", "^", "D"]
    colores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    max_p_global = max(r["hilos"] for r in registros)
    p_continuo = np.linspace(1.0, max(max_p_global, 2), 100)

    # Línea ideal en Speedup y Eficiencia
    axes[1].plot(p_continuo, p_continuo, "k--", linewidth=1.5, label="Speedup ideal ($S_p = p$)")
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1.5, label="Eficiencia ideal ($E_p = 1$)")

    for idx, (alg, n) in enumerate(claves):
        sub = [r for r in registros if r["algoritmo"] == alg and r["tamano_malla_N"] == n and r["version"] == "paralelo"]
        sub.sort(key=lambda item: item["hilos"])

        p_vals = np.array([r["hilos"] for r in sub], dtype=np.int32)
        t_vals = np.array([r["tiempo_mediana_s"] for r in sub], dtype=np.float64)
        std_vals = np.array([r["tiempo_std_s"] for r in sub], dtype=np.float64)
        s_vals = np.array([r["speedup"] for r in sub], dtype=np.float64)
        e_vals = np.array([r["eficiencia"] for r in sub], dtype=np.float64)

        f_amdahl = estimar_fraccion_amdahl(p_vals, s_vals)
        etiqueta_corta = "Coulomb" if "Coulomb" in alg else "Legendre"
        color = colores[idx % len(colores)]
        marcador = marcadores[idx % len(marcadores)]

        # Panel 1: Tiempo vs Hilos (con barras de dispersión)
        axes[0].errorbar(
            p_vals,
            t_vals,
            yerr=std_vals,
            fmt=f"{marcador}-",
            color=color,
            capsize=4,
            linewidth=1.8,
            label=f"{etiqueta_corta} ($N={n}^2$)",
        )

        # Panel 2: Speedup vs Hilos + Ajuste de Amdahl
        axes[1].plot(
            p_vals,
            s_vals,
            f"{marcador}-",
            color=color,
            linewidth=1.8,
            label=f"{etiqueta_corta} ($N={n}^2$, $f\\approx{f_amdahl:.3f}$)",
        )
        s_amdahl_curva = 1.0 / ((1.0 - f_amdahl) + (f_amdahl / p_continuo))
        axes[1].plot(p_continuo, s_amdahl_curva, ":", color=color, alpha=0.75, linewidth=1.3)

        # Panel 3: Eficiencia vs Hilos
        axes[2].plot(
            p_vals,
            e_vals,
            f"{marcador}-",
            color=color,
            linewidth=1.8,
            label=f"{etiqueta_corta} ($N={n}^2$)",
        )

    # Formato Panel 1 (Tiempo)
    axes[0].set_title("Tiempo de Cómputo vs. Hilos (Mediana $\\pm \\sigma$)")
    axes[0].set_xlabel("Número de hilos ($p$)")
    axes[0].set_ylabel("Tiempo de ejecución $T_p$ [s]")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Formato Panel 2 (Speedup)
    axes[1].set_title("Speedup $S_p = T_1 / T_p$ y Ajuste Ley de Amdahl")
    axes[1].set_xlabel("Número de hilos ($p$)")
    axes[1].set_ylabel("Speedup $S_p$")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5, loc="upper left")

    # Formato Panel 3 (Eficiencia)
    axes[2].set_title("Eficiencia Paralela $E_p = S_p / p$")
    axes[2].set_xlabel("Número de hilos ($p$)")
    axes[2].set_ylabel("Eficiencia $E_p$")
    axes[2].set_ylim(0.0, 1.15)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=9, loc="lower left")

    fig.savefig(ruta_salida, dpi=200)
    plt.close(fig)
    print(f"Gráfica de desempeño guardada en '{ruta_salida}'.")


def analizar_colapso_convergencia(
    a: float = 1.0,
    theta_rad: float = np.pi / 3.0,
    ruta_csv: str = "colapso_convergencia.csv",
    ruta_figura: str = "colapso_convergencia_r_a.png",
):
    """
    Evalúa cuantitativamente el colapso de convergencia de la serie multipolar de Legendre
    cuando r se aproxima al radio del anillo a (r / a -> 1), midiendo cómo crecen el número
    de términos requeridos, el error numérico y el costo computacional.
    """
    radios_relativos = [0.50, 0.80, 0.95, 0.99, 1.00]
    lista_k = [2, 5, 10, 20, 35, 60, 100, 180, 300, 500]
    m_ref = 2000  # Cuadratura de alta precisión como referencia exacta

    filas_csv = []
    curvas_error = {r_rel: [] for r_rel in radios_relativos}

    for r_rel in radios_relativos:
        r_val = r_rel * a
        x_pt = r_val * np.sin(theta_rad)
        z_pt = r_val * np.cos(theta_rad)
        v_exacto = _potencial_coulomb_punto_serial(x_pt, z_pt, a, m_ref)

        for k in lista_k:
            v_serie = _potencial_legendre_punto_serial(x_pt, z_pt, a, k)
            err_rel = abs(v_serie - v_exacto) / max(abs(v_exacto), 1e-16)
            curvas_error[r_rel].append(max(err_rel, 1e-16))
            filas_csv.append({
                "r_sobre_a": r_rel,
                "theta_grados": np.degrees(theta_rad),
                "n_terminos_K": k,
                "v_coulomb_ref": v_exacto,
                "v_legendre": v_serie,
                "error_relativo": err_rel,
            })

    # Medir el tiempo de cómputo en una malla 400x400 en función de K frente a Coulomb (M=200)
    x_m, z_m = crear_malla(400, 2.0)
    t0 = time.perf_counter()
    _ = calcular_potencial_coulomb_serial(x_m, z_m, a, 200)
    t_coulomb_fijo = time.perf_counter() - t0

    tiempos_k = []
    for k in lista_k:
        t0 = time.perf_counter()
        _ = calcular_potencial_legendre_serial(x_m, z_m, a, k)
        tiempos_k.append(time.perf_counter() - t0)

    # Guardar CSV
    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(filas_csv[0].keys()))
        writer.writeheader()
        writer.writerows(filas_csv)

    # Graficar
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    for r_rel in radios_relativos:
        estilo = "s--" if r_rel == 1.00 else "o-"
        axes[0].plot(
            lista_k,
            curvas_error[r_rel],
            estilo,
            linewidth=1.8,
            label=f"$r / a = {r_rel:.2f}$",
        )

    axes[0].axhline(2.22e-16, color="gray", linestyle=":", label="Épsilon de máquina ($\\approx 2.2\\times 10^{-16}$)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Convergencia Multipolar al Acercarse a $r \\approx a$ ($\\theta = 60^\\circ$)")
    axes[0].set_xlabel("Número de términos multipolares ($K$)")
    axes[0].set_ylabel("Error relativo $|V_{\\mathrm{serie}} - V_{\\mathrm{num}}| / V_{\\mathrm{num}}$")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    axes[1].plot(lista_k, tiempos_k, "o-", color="#d62728", linewidth=2.0, label="Serie de Legendre ($K$ términos)")
    axes[1].axhline(
        t_coulomb_fijo,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        label="Integración de Coulomb ($M=200$ pasos)",
    )
    axes[1].set_title("Cruce de Costo Computacional en Malla $400 \\times 400$ (Serial)")
    axes[1].set_xlabel("Número de términos multipolares ($K$)")
    axes[1].set_ylabel("Tiempo de ejecución [s]")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9.5)

    fig.savefig(ruta_figura, dpi=200)
    plt.close(fig)
    print(f"Análisis de colapso en r ≈ a guardado en '{ruta_csv}' y '{ruta_figura}'.")


if __name__ == "__main__":
    # 1. Reporte de hardware
    reportar_hardware()

    # 2. Medición aislada de la compilación JIT
    medir_compilacion_jit()

    # 3. Verificación cuantitativa Serial vs. Paralelo (Criterio 4)
    verificar_equivalencia_serial_paralelo(tamanos_malla=(200, 500, 800))

    # 4. Regenerar Figura 1: Potencial físico, campo eléctrico y mapa de error 2D
    print("\n=== Generando Figura 1: Visualización física del potencial y colapso en r ≈ a ===")
    x_vis, z_vis = crear_malla(n_puntos=400, limite=2.0)
    v_coul_vis = calcular_potencial_coulomb_serial(x_vis, z_vis, a=1.0, m_pasos=200)
    v_leg_vis = calcular_potencial_legendre_serial(x_vis, z_vis, a=1.0, n_terminos=25)
    graficar_fenomeno_y_error(
        x_vis,
        z_vis,
        v_coul_vis,
        v_leg_vis,
        a=1.0,
        n_terminos=25,
        ruta_salida="verificacion_fisica_anillo.png",
    )
    print("Figura 1 guardada en 'verificacion_fisica_anillo.png'.\n")

    # 5. Ejecutar benchmark de tiempos, Speedup y Eficiencia (Criterios 5 y 6)
    datos_benchmark = ejecutar_benchmark_desempeno(
        tamanos_malla=(600, 1500),
        repeticiones=5,
        m_pasos=200,
        k_terminos=60,
        a=1.0,
        ruta_csv="resultados_desempeno.csv",
    )
    graficar_resultados_desempeno(
        datos_benchmark,
        ruta_salida="desempeno_speedup_eficiencia.png",
    )

    # 6. Ejecutar análisis del colapso computacional en r ≈ a
    print("\n=== Generando Figura 3: Colapso de convergencia y cruce de tiempos en r ≈ a ===")
    analizar_colapso_convergencia(
        a=1.0,
        theta_rad=np.pi / 3.0,
        ruta_csv="colapso_convergencia.csv",
        ruta_figura="colapso_convergencia_r_a.png",
    )
    print("\n¡Proceso completo! Todos los archivos CSV y figuras han sido regenerados.")
    