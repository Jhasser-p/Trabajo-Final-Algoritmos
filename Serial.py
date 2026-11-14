"""
serial.py
Implementación serial del potencial electrostático y campo eléctrico de un anillo cargado.
Incluye:
1. Integración numérica directa (Ley de Coulomb mediante regla de Simpson compuesta).
2. Expansión multipolar en Polinomios de Legendre (recurrencia de Bonnet).
3. Visualización física del potencial, campo eléctrico y colapso en r ≈ a.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def crear_malla(n_puntos: int = 500, limite: float = 2.0):
    """
    Inicializa la malla 2D en el plano xz (con y = 0 por simetría azimutal).
    Se desplaza ligeramente la malla si cae exactamente en (x=1, z=0) para evitar
    la singularidad matemática sobre el alambre del anillo (r = a, theta = pi/2).
    """
    x = np.linspace(-limite, limite, n_puntos, dtype=np.float64)
    z = np.linspace(-limite, limite, n_puntos, dtype=np.float64)
    return x, z


@njit(fastmath=False)
def _potencial_coulomb_punto_serial(x_val: float, z_val: float, a: float, m_pasos: int) -> float:
    """
    Evalúa la integral elíptica de Coulomb en un punto (x, z) usando la regla de Simpson 1/3.
    V(x, z) = (1 / pi) * int_0^pi dphi' / sqrt(x^2 + z^2 + a^2 - 2*a*x*cos(phi'))
    Se asegura que m_pasos sea par.
    """
    if m_pasos % 2 != 0:
        m_pasos += 1

    h = np.pi / m_pasos
    r2_a2 = x_val * x_val + z_val * z_val + a * a
    dos_ax = 2.0 * a * x_val

    # Extremos phi' = 0 y phi' = pi
    d0 = r2_a2 - dos_ax
    dpi = r2_a2 + dos_ax
    if d0 < 1e-18:
        d0 = 1e-18
    if dpi < 1e-18:
        dpi = 1e-18

    suma = (1.0 / np.sqrt(d0)) + (1.0 / np.sqrt(dpi))

    for k in range(1, m_pasos):
        phi = k * h
        denom = r2_a2 - dos_ax * np.cos(phi)
        if denom < 1e-18:
            denom = 1e-18
        peso = 4.0 if (k % 2 == 1) else 2.0
        suma += peso / np.sqrt(denom)

    return (h / (3.0 * np.pi)) * suma


@njit(fastmath=False)
def calcular_potencial_coulomb_serial(x: np.ndarray, z: np.ndarray, a: float = 1.0, m_pasos: int = 200) -> np.ndarray:
    """
    Calcula en forma serial (1 hilo) el potencial electrostático en toda la malla N x N
    mediante integración numérica directa de la Ley de Coulomb.
    """
    nx = x.shape[0]
    nz = z.shape[0]
    potencial = np.empty((nz, nx), dtype=np.float64)

    for i in range(nz):
        z_val = z[i]
        for j in range(nx):
            x_val = x[j]
            potencial[i, j] = _potencial_coulomb_punto_serial(x_val, z_val, a, m_pasos)

    return potencial


@njit(fastmath=False)
def _potencial_legendre_punto_serial(x_val: float, z_val: float, a: float, n_terminos: int) -> float:
    """
    Evalúa la expansión multipolar en Polinomios de Legendre en (x, z) sumando los primeros
    n_terminos multipolos pares (l = 2k, con k = 0, 1, ..., n_terminos - 1).
    Usa la relación de recurrencia de Bonnet para generar P_l(cos(theta)) sin llamadas externas.
    """
    r = np.sqrt(x_val * x_val + z_val * z_val)
    if r < 1e-15:
        # En el origen r = 0, solo sobrevive el término monopolar k = 0: V(0) = 1 / a
        return 1.0 / a

    u = z_val / r  # cos(theta)
    if r < a:
        r_menor = r
        r_mayor = a
    else:
        r_menor = a
        r_mayor = r

    razon = r_menor / r_mayor
    razon_cuad = razon * razon

    # Término k = 0 (l = 0, monopolo): P_0(0) = 1, P_0(u) = 1
    p_lm2 = 1.0  # P_0(u)
    p_lm1 = u    # P_1(u)

    coef_ecuatorial = 1.0  # P_{2k}(0) = (-1)^k (2k)! / (2^{2k} (k!)^2)
    potencia_r = 1.0 / r_mayor

    suma = coef_ecuatorial * potencia_r * p_lm2

    for k in range(1, n_terminos):
        # Avanzamos dos pasos en la recurrencia de Bonnet para obtener P_{2k}(u)
        l_impar = 2 * k - 1
        # P_{2k-1}(u) ya está en p_lm1 para k=1; para k > 1 lo calculamos en el paso anterior
        # Paso par: l = 2k
        l_par = 2 * k
        p_l_par = ((2.0 * l_par - 1.0) * u * p_lm1 - (l_par - 1.0) * p_lm2) / l_par

        # Actualizamos coeficiente ecuatorial P_{2k}(0) = - P_{2k-2}(0) * (2k - 1) / (2k)
        coef_ecuatorial = -coef_ecuatorial * (2.0 * k - 1.0) / (2.0 * k)
        potencia_r *= razon_cuad

        suma += coef_ecuatorial * potencia_r * p_l_par

        # Preparamos el siguiente impar P_{2k+1}(u) si quedan iteraciones
        if k + 1 < n_terminos:
            l_sig_impar = 2 * k + 1
            p_l_impar = ((2.0 * l_sig_impar - 1.0) * u * p_l_par - (l_sig_impar - 1.0) * p_lm1) / l_sig_impar
            p_lm2 = p_l_par
            p_lm1 = p_l_impar

    return suma


@njit(fastmath=False)
def calcular_potencial_legendre_serial(x: np.ndarray, z: np.ndarray, a: float = 1.0, n_terminos: int = 40) -> np.ndarray:
    """
    Calcula en forma serial (1 hilo) el potencial electrostático en toda la malla N x N
    mediante la serie multipolar de Polinomios de Legendre.
    """
    nx = x.shape[0]
    nz = z.shape[0]
    potencial = np.empty((nz, nx), dtype=np.float64)

    for i in range(nz):
        z_val = z[i]
        for j in range(nx):
            x_val = x[j]
            potencial[i, j] = _potencial_legendre_punto_serial(x_val, z_val, a, n_terminos)

    return potencial


def calcular_campo_electrico(potencial: np.ndarray, x: np.ndarray, z: np.ndarray):
    """
    Calcula el campo eléctrico E = -grad(V) mediante diferencias finitas centradas (np.gradient).
    Retorna (Ex, Ez).
    """
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    grad_z, grad_x = np.gradient(potencial, dz, dx)
    return -grad_x, -grad_z

def graficar_fenomeno_y_error(
    x: np.ndarray,
    z: np.ndarray,
    v_coulomb: np.ndarray,
    v_legendre: np.ndarray,
    a: float = 1.0,
    n_terminos: int = 25,
    ruta_salida: str = "verificacion_fisica_anillo.png",
    mostrar: bool = True,
):
    """
    Genera una figura compuesta con:
    1. Mapa de potencial electrostático y líneas de campo eléctrico (Coulomb).
    2. Mapa de error relativo logarítmico |V_num - V_serie| / |V_num| mostrando el colapso en r = a.
    """
    ex, ez = calcular_campo_electrico(v_coulomb, x, z)
    error_rel = np.abs(v_coulomb - v_legendre) / np.maximum(np.abs(v_coulomb), 1e-15)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # Panel 1: Potencial y líneas de campo
    v_recortado = np.clip(v_coulomb, 0.0, 3.5)
    im0 = axes[0].contourf(x, z, v_recortado, levels=40, cmap="inferno")
    axes[0].streamplot(x, z, ex, ez, color="white", linewidth=0.7, density=1.2, arrowsize=0.9)
    axes[0].plot([-a, a], [0.0, 0.0], "co", markersize=8, markeredgecolor="black", label=r"Corte del anillo ($r=a$)")
    axes[0].set_title(r"Potencial $V(x,z)$ y Campo $\mathbf{E}$ (Integración de Coulomb)")
    axes[0].set_xlabel(r"Posición $x / a$")
    axes[0].set_ylabel(r"Posición $z / a$")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label(r"Potencial adimensional $V / (Q / 4\pi\varepsilon_0 a)$")

    # Panel 2: Colapso en el borde del radio de convergencia r = a
    log_err = np.log10(np.maximum(error_rel, 1e-16))
    im1 = axes[1].pcolormesh(x, z, log_err, shading="auto", cmap="viridis", vmin=-14, vmax=0)
    theta_circ = np.linspace(0, 2 * np.pi, 300)
    axes[1].plot(a * np.cos(theta_circ), a * np.sin(theta_circ), "r--", linewidth=1.5, label=r"Frontera $r = a$")
    axes[1].set_title(rf"Colapso de convergencia en $r \approx a$ ($K={n_terminos}$ términos)")
    axes[1].set_xlabel(r"Posición $x / a$")
    axes[1].set_ylabel(r"Posición $z / a$")
    axes[1].set_aspect("equal")
    axes[1].legend(loc="upper right")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label(r"$\log_{10}(|V_{\mathrm{num}} - V_{\mathrm{serie}}| / V_{\mathrm{num}})$")

    # 1. Guardar en disco en alta resolución
    fig.savefig(ruta_salida, dpi=200)
    print(f"Figura guardada en '{ruta_salida}'.")

    # 2. Mostrar en ventana interactiva
    if mostrar:
        plt.show()

    # 3. Liberar memoria
    plt.close(fig)

if __name__ == "__main__":
    # Parámetros de prueba inicial
    N = 400
    M_PASOS = 200
    K_TERMINOS = 25
    RADIO_A = 1.0

    x_grid, z_grid = crear_malla(n_puntos=N, limite=2.0)

    # Calentamiento JIT (compilación aislada del tiempo de cómputo)
    _ = calcular_potencial_coulomb_serial(x_grid[:8], z_grid[:8], RADIO_A, M_PASOS)
    _ = calcular_potencial_legendre_serial(x_grid[:8], z_grid[:8], RADIO_A, K_TERMINOS)

    t0 = time.perf_counter()
    v_num = calcular_potencial_coulomb_serial(x_grid, z_grid, RADIO_A, M_PASOS)
    t_coulomb = time.perf_counter() - t0

    t0 = time.perf_counter()
    v_leg = calcular_potencial_legendre_serial(x_grid, z_grid, RADIO_A, K_TERMINOS)
    t_legendre = time.perf_counter() - t0

    print(f"[Serial] Malla {N}x{N} | Tiempo Coulomb (M={M_PASOS}): {t_coulomb:.4f} s")
    print(f"[Serial] Malla {N}x{N} | Tiempo Legendre (K={K_TERMINOS}): {t_legendre:.4f} s")

    graficar_fenomeno_y_error(x_grid, z_grid, v_num, v_leg, a=RADIO_A, n_terminos=K_TERMINOS)
    print("Figura guardada en 'verificacion_fisica_anillo.png'.")