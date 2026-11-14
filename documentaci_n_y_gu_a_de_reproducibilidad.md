# Simulación Serial y Paralela del Potencial Electrostático de un Anillo de Carga: Integración de Coulomb vs. Expansión Multipolar de Legendre
**Samuel Cala-2222222**  
**Joan Rivero-2222222**  
**Nicolas DE Jesus Gamboa-2222222**   
**Jhasser Andres Piña Peña-2251982**  
**Universidad Industrial de Santander (UIS)**  
**Escuela de Física — Cálculo Científico (2026-2)**

---

## 1. Descripción del Proyecto

Este proyecto simula el potencial electrostático $V(x,z)$ y el campo eléctrico $\mathbf{E}(x,z) = -\nabla V(x,z)$ generados por un anillo circular de radio $a$ con densidad lineal de carga uniforme fuera de su eje de simetría, donde no existe solución analítica en términos de funciones elementales.

Se implementan y paralelizan en CPU (mediante `numba.prange`) dos formulaciones matemáticas complementarias sobre una malla bidimensional de $N \times N$ nodos:

1. **Integración Numérica Directa (Ley de Coulomb):** Evaluación de la integral elíptica mediante la regla de Simpson $1/3$ compuesta de $M$ subintervalos en cada coordenada espacial:
   $$V_{\text{num}}(x, z) = \frac{1}{\pi} \int_0^{\pi} \frac{d\phi'}{\sqrt{x^2 + z^2 + a^2 - 2ax\cos\phi'}}$$
2. **Expansión Multipolar en Polinomios de Legendre (Arfken, Cap. 5):** Desarrollo en serie de potencias a partir de la función generatriz $(1 - 2ut + t^2)^{-1/2}$ diferenciando las regiones de convergencia interior ($r < a$) y exterior ($r > a$):
   $$V_{\text{serie}}(r, \theta) = \sum_{k=0}^{K-1} (-1)^k \frac{(2k)!}{2^{2k}(k!)^2} \left(\frac{r_<^{2k}}{r_>^{2k+1}}\right) P_{2k}(\cos\theta)$$
   donde $r_< = \min(r, a)$, $r_> = \max(r, a)$ y los polinomios $P_l(\cos\theta)$ se evalúan en línea dentro del kernel compilado usando la relación de recurrencia de Bonnet.

---

## 2. Estructura del Repositorio

* `serial.py`: Implementación secuencial (1 hilo) de ambos algoritmos, cálculo del campo eléctrico por diferencias finitas centradas y visualización física del potencial y del borde de convergencia $r = a$.
* `paralelo.py`: Implementación paralela por hilos en CPU utilizando `@njit(parallel=True)` y `prange` sobre el bucle espacial externo, evitando condiciones de carrera y liberando el GIL. Incluye la rutina de verificación cuantitativa (`np.allclose` y diferencia absoluta máxima).
* `medir.py`: Script automatizado que reporta el hardware, aísla el tiempo de compilación JIT, ejecuta los experimentos de desempeño ($\ge 5$ repeticiones por configuración con `time.perf_counter`), estima la fracción paralelizable $f$ de la Ley de Amdahl, exporta los archivos CSV y regenera todas las figuras del proyecto.
* `requirements.txt`: Lista de librerías necesarias con sus versiones mínimas.

---

## 3. Entorno de Software y Hardware Utilizado

### Software requerido
* **Python:** `>= 3.10` (probado en Python `3.11` / `3.12`)
* **Librerías principales:** `numpy`, `numba`, `matplotlib`

### Hardware de referencia
Al ejecutar `python medir.py`, el script imprime en consola las especificaciones exactas del equipo donde se corre la prueba. Para documentar en el informe tu propio equipo, verifica el bloque `ESPECIFICACIONES DEL ENTORNO Y HARDWARE` que genera el script al inicio:
* **Arquitectura:** `x86_64` / `ARM64`
* **Paralelismo:** Hilos de CPU gestionados por la capa de paralelismo de Numba (`numba.set_num_threads(p)`).

---

## 4. Instrucciones de Instalación y Reproducibilidad

Sigue estos pasos para instalar las dependencias y regenerar todos los datos y figuras desde cero:

### Paso 1: Crear un entorno virtual e instalar dependencias
```bash
python -m venv venv
# En Linux / macOS:
source venv/bin/activate
# En Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Paso 2: Ejecutar la verificación numérica rápida (Criterio 4)
Para comprobar que las versiones serial y paralela producen exactamente la misma matriz numérica dentro de la tolerancia de punto flotante de doble precisión (`float64`):
```bash
python paralelo.py
```

### Paso 3: Regenerar todos los CSV y todas las figuras del proyecto (Criterios 5, 6 y 7)
Ejecuta el script maestro de medición:
```bash
python medir.py
```
Este único comando genera automáticamente los siguientes archivos en el directorio raíz:
1. **`resultados_desempeno.csv`**: Tabla completa con tiempos (mediana, promedio, desviación estándar $\sigma$, rango intercuartílico IQR, mínimo y máximo), *speedup* ($S_p = T_1 / T_p$) y eficiencia ($E_p = S_p / p$) variando el número de hilos en dos tamaños de malla ($600 \times 600$ y $1500 \times 1500$).
2. **`colapso_convergencia.csv`**: Datos del error relativo y costo computacional de la serie de Legendre en función del número de términos $K$ a medida que $r / a \to 1$.
3. **`verificacion_fisica_anillo.png` (Figura 1)**: Mapa de contorno del potencial electrostático $V(x,z)$, líneas de corriente del campo eléctrico $\mathbf{E}$ y mapa logarítmico del error relativo mostrando la divergencia sobre la circunferencia $r = a$.
4. **`desempeno_speedup_eficiencia.png` (Figura 2)**: Panel triple con Tiempo de Ejecución vs. Hilos, *Speedup* vs. Hilos (incluyendo la recta ideal $S_p = p$ y la curva ajustada de la Ley de Amdahl con la fracción paralelizable $f$) y Eficiencia Paralela vs. Hilos.
5. **`colapso_convergencia_r_a.png` (Figura 3)**: Curvas de decaimiento del error relativo para distintas razones $r/a \in [0.50, 1.00]$ y gráfica del cruce de tiempo de cómputo entre la serie multipolar y la cuadratura de Simpson cuando $K$ crece.