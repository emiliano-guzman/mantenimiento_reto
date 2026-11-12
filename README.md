# Reto de mantenimiento predictivo

Preparación de datos para anticipar si una máquina tendrá una falla en las siguientes 24 horas y qué componente fallará. Se utilizan los cinco archivos del conjunto Microsoft Azure Predictive Maintenance.

## Estado del proyecto

Ya están implementados la revisión de datos, el análisis exploratorio, la relación entre mantenimiento y fallas, y el pipeline que integra variables históricas, etiquetas y partición temporal.

### Cierre de la primera entrega

La primera fase queda identificada con el tag **`entrega1`**, que conserva la versión del código correspondiente al reporte: revisión y EDA, análisis de mantenimiento, variables históricas, etiquetas, partición temporal y comparación de regresión logística y XGBoost en validación. Los resultados están en `Modelación/05_línea_base_clásica.ipynb`; el reporte se entrega por separado.

La versión entregada puede consultarse en [GitHub: entrega1](https://github.com/emiliano-guzman/mantenimiento_reto/tree/entrega1). El trabajo de las siguientes fases podrá continuar en `main` sin mover este tag. La evaluación de modelos sobre prueba sigue reservada para la tercera semana.

Para recuperar esta versión en una copia del repositorio:

```bash
git fetch --tags
git switch --detach entrega1
```

Este comando deja la copia en la versión entregada para consultarla o reproducirla. Para volver a la rama de trabajo: `git switch main`. Ejecutar los cambios de versión con el trabajo local previamente guardado en Git o resguardado.


## Organización

```text
Datos/
├── PdM_telemetry.csv
├── PdM_errors.csv
├── PdM_maint.csv
├── PdM_failures.csv
├── PdM_machines.csv
└── Datos_preparados/             # Se genera localmente
    ├── entrenamiento.csv
    ├── validación.csv
    └── prueba.csv
EDA/
├── 02_análisis_exploratorio.ipynb
└── Figuras/
Preprocesamiento/
├── 01_revisión_datos.ipynb
├── 03_mantenimiento_fallas.ipynb
├── 04_preparación_datos.ipynb
└── preparar_datos.py
Modelación/
└── 05_línea_base_clásica.ipynb
requirements.txt
```

**Los tres CSV preparados no se subieron a GitHub.** Se regeneran a partir de los cinco CSV originales mediante el notebook 04 o el script. No es necesario instalar Git LFS para reproducir el proyecto. El entorno virtual y `__pycache__` tampoco se versionan; Python puede volver a crear esa caché al importar el script.

## Entorno de Python

Usar Python 3.12. El entorno virtual es local y debe crearse **fuera del repositorio**; no se sube a GitHub. El archivo `requirements.txt` registra las versiones instaladas de las dependencias.

En macOS, instalar primero OpenMP (requerido por XGBoost y LightGBM) con [Homebrew](https://brew.sh/):

```bash
brew install libomp
```

En macOS o Linux, desde la carpeta del repositorio:

```bash
python3.12 -m venv ../reto-nov
source ../reto-nov/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Si tu comando `python3` corresponde a Python 3.12, puedes usarlo en lugar de `python3.12`.

Para abrir notebooks:

```bash
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name reto-nov --display-name "Python (reto-nov)"
python -m jupyterlab
```

En VS Code, seleccionar como intérprete o kernel el ejecutable `bin/python` del entorno `reto-nov`.

Para salir del entorno:

```bash
deactivate
```

## Dependencias

- Datos y gráficas: NumPy, pandas, SciPy, Matplotlib y Seaborn.
- Modelos clásicos y evaluación: scikit-learn, XGBoost y LightGBM.
- Deep learning: PyTorch.
- Optimización y desbalance: Optuna e imbalanced-learn.
- Interpretabilidad: SHAP y Captum.
- Notebooks y utilidades: JupyterLab, ipykernel, widgets, PyArrow, PyYAML, joblib, tqdm y TensorBoard.
- Verificación: pytest.

Las dependencias se validan inicialmente en macOS con Apple Silicon y Python 3.12. La disponibilidad de aceleración de PyTorch depende del equipo; se puede ejecutar en CPU.

### Uso conjunto en macOS

En la configuración local se comprobó un conflicto de OpenMP al cargar PyTorch antes de XGBoost. Si un notebook usa ambas familias de modelos, importar primero XGBoost y LightGBM:

```python
import xgboost
import lightgbm
import torch
```

Se verificó el entrenamiento de regresión logística, XGBoost y LightGBM, además de una pasada hacia adelante y hacia atrás de una LSTM de PyTorch, con datos sintéticos. El uso conjunto de XGBoost, LightGBM y PyTorch se validó con el orden anterior. Si ya se importó PyTorch primero, reiniciar el kernel antes de cambiar el orden.

## Regenerar los conjuntos preparados

Primero instalar las dependencias y activar el entorno. Los cinco CSV originales deben estar en `Datos/`, con los nombres indicados arriba. Los comandos siguientes se ejecutan desde la raíz del repositorio.

### Opción 1: ejecutar el pipeline completo

```bash
python Preprocesamiento/preparar_datos.py
```

### Opción 2: ejecutar el notebook explicado

Abrir `Preprocesamiento/04_preparación_datos.ipynb` y ejecutar todas las celdas en orden. Puede abrirse desde la raíz del proyecto o desde `Preprocesamiento`. Este notebook importa las mismas funciones del script, por lo que ambas opciones realizan la misma preparación; basta con ejecutar una.

No es necesario correr los notebooks de revisión, EDA o mantenimiento para generar los conjuntos. Esos notebooks explican los análisis previos; el pipeline lee directamente los originales.

Ambas opciones crean automáticamente `Datos/Datos_preparados/` y guardan:

- `entrenamiento.csv`
- `validación.csv`
- `prueba.csv`

Al repetir la ejecución se reemplazan esos tres archivos generados. Los cinco CSV originales permanecen intactos. Cada conjunto contiene **35 columnas: dos claves (`machineID`, `datetime`), 31 variables de entrada y dos etiquetas**. Los archivos preparados ocupan aproximadamente 363 MB en total con los datos actuales.

## Qué hace el pipeline

1. Lee los cinco CSV originales y comprueba sus claves.
2. Usa la telemetría como base por máquina y hora. Calcula medias y desviaciones estándar de cada sensor en ventanas de 3 y 24 horas.
3. Agrega cinco conteos de errores en las últimas 24 horas, cuatro tiempos desde reemplazos y el modelo y antigüedad del catálogo. Conserva también los cuatro sensores originales.
4. Separa las horas por periodo y retira las que no tienen su horizonte completo dentro del conjunto.
5. Agrega `etiqueta_binaria` y `etiqueta_multiclase` usando las fallas del periodo correspondiente.
6. Guarda los tres conjuntos completos, con variables y respuestas juntas.

Las variables se calculan por máquina antes de separar periodos, para que septiembre pueda aprovechar historia de agosto. Los sensores y los errores usan ventanas históricas que incluyen la hora actual, suponiendo que ya se observó su información. Los reemplazos se consultan estrictamente antes de la hora actual; también se utiliza el historial disponible de 2014.

Las primeras ventanas de sensores que no tienen 3 o 24 mediciones completas quedan vacías. Si no existe un reemplazo anterior conocido, su tiempo también queda vacío. No se rellena con información futura. La ausencia de registros de errores se cuenta como cero, suponiendo un registro completo. La imputación de faltantes y la codificación se ajustan solo con entrenamiento dentro de los pipelines de modelado del notebook 05. La regresión logística también estandariza las variables numéricas; XGBoost no requiere ese escalado. Los CSV preparados se conservan sin estas transformaciones específicas del modelo.

### Etiquetas y partición temporal

La etiqueta binaria vale 1 si ocurre alguna falla en **(t, t + 24 h]**, y 0 si no ocurre ninguna. La multiclase contiene `ninguno`, `comp1`, `comp2`, `comp3` o `comp4`. Si hay varias fallas, se elige la primera; si esa primera hora incluye varios componentes, se aplica el orden comp1–comp4 como desempate fijo. No representa una prioridad de gravedad y puede perder información sobre fallas simultáneas.

| Conjunto | Periodo establecido | Última hora conservada con el margen |
|---|---|---|
| Entrenamiento | Enero–agosto de 2015 | 30 de agosto a las 23:00 |
| Validación | Septiembre–octubre de 2015 | 30 de octubre a las 23:00 |
| Prueba | Noviembre–diciembre de 2015 | 30 de diciembre a las 23:00 |

Se exige que `t + 24 h` sea anterior al límite excluido de cada periodo. Así ninguna etiqueta de entrenamiento depende de septiembre ni una de validación de noviembre. En diciembre se retiran también las últimas 24 horas para no asumir seguimiento completo de fallas en 2016. Con los originales actuales se obtienen 580,200 filas de entrenamiento, 144,000 de validación y 144,000 de prueba.

**Prueba se prepara y guarda, pero su evaluación queda reservada para la semana 3**, con el modelo y umbral ya elegidos. Los análisis de desbalance del notebook se limitan a entrenamiento y validación. No se utiliza prueba para seleccionar variables, modelos, hiperparámetros o umbrales.

Las dos etiquetas son respuestas futuras, no variables de entrada. `machineID` y `datetime` son claves de identificación y tampoco se incorporan automáticamente como predictores.

## Comprobaciones y reproducibilidad

El notebook 04 contrasta ejemplos de entrenamiento y validación con cálculos directos desde las fuentes: ventanas de sensores, conteos de errores, reemplazos y etiquetas. También verifica las claves, las fronteras y la estructura de los CSV exportados. El script incluye comprobaciones de integridad y separación temporal.

La preparación actual no usa muestreo ni operaciones aleatorias: con los mismos CSV, dependencias y código se esperan los mismos conjuntos. El notebook 05 fija la semilla 42 y los parámetros de los modelos. Al ejecutar sus celdas se regeneran las métricas de validación, las curvas y las matrices multiclase dentro del propio notebook, sin exportaciones adicionales. Estos son resultados de desarrollo; aún no hay evaluación final en prueba.
