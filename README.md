# Reto de mantenimiento predictivo

Preparación de datos para anticipar si una máquina tendrá una falla en las siguientes 24 horas y qué componente fallará. Se utilizan los cinco archivos del conjunto Microsoft Azure Predictive Maintenance.

## Estado del proyecto

El proyecto incluye la preparación de datos, la línea base clásica y la comparación de modelos secuenciales. La segunda entrega añade LSTM, TCN y CNN-LSTM con dos salidas, comparación de estrategias de desbalance, curvas de aprendizaje y evaluación de alertas sostenidas en validación. El conjunto de prueba todavía no se ha evaluado.

### Cierre de la primera entrega

La primera fase queda identificada con el tag **`entrega1`**, que conserva la versión del código correspondiente al reporte: revisión y EDA, análisis de mantenimiento, variables históricas, etiquetas, partición temporal y comparación de regresión logística y XGBoost en validación. Los resultados están en `Modelación/05_línea_base_clásica.ipynb`; el reporte se entrega por separado.

La versión entregada puede consultarse en [GitHub: entrega1](https://github.com/emiliano-guzman/mantenimiento_reto/tree/entrega1). El trabajo de las siguientes fases podrá continuar en `main` sin mover este tag. La evaluación de modelos sobre prueba sigue reservada para la tercera semana.

Para recuperar esta versión en una copia del repositorio:

```bash
git fetch --tags
git switch --detach entrega1
```

Este comando deja la copia en la versión entregada para consultarla o reproducirla. Para volver a la rama de trabajo: `git switch main`. Ejecutar los cambios de versión con el trabajo local previamente guardado en Git o resguardado.


### Cierre de la segunda entrega

El tag **`entrega2`** conserva el código correspondiente a la segunda fase: preparación de ventanas, arquitecturas de dos salidas, desbalance y entrenamiento disciplinado, comparación con la línea base, F1 por componente y anticipación de alertas sostenidas. Los procedimientos y resultados están en los notebooks 06–09; el segundo reporte se entrega por separado.

La versión puede consultarse en [GitHub: entrega2](https://github.com/emiliano-guzman/mantenimiento_reto/tree/entrega2). Este tag y `entrega1` se conservan fijos; las siguientes fases continúan en `main`.

Para recuperar la segunda entrega:

```bash
git fetch --tags
git switch --detach entrega2
```

Para volver a la rama de trabajo: `git switch main`. Guardar o resguardar los cambios locales antes de cambiar de versión. Prueba sigue reservada para la tercera entrega, con el modelo y la política ya elegidos.


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
├── 06_preparación_secuencias.ipynb
├── preparar_datos.py
└── preparar_secuencias.py
Modelación/
├── 05_línea_base_clásica.ipynb
├── 07_arquitecturas_secuenciales.ipynb
├── 08_desbalance_y_entrenamiento.ipynb
├── 09_comparación_y_anticipación.ipynb
├── arquitecturas_secuenciales.py
└── entrenamiento_secuencial.py
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

### PyTorch y XGBoost en macOS

En el entorno local se observó un conflicto de bibliotecas nativas al entrenar XGBoost después de cargar PyTorch en el mismo proceso. Por eso el notebook 09 reproduce la línea base en un proceso separado, usando el mismo Python del entorno activo. Intercambia únicamente las probabilidades mediante un archivo temporal que se elimina automáticamente.

Este aislamiento ya está incluido en el notebook: no hay que ejecutar un programa adicional ni conservar ese archivo temporal. El notebook 05 puede ejecutarse por separado en su propio kernel.

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

## Reproducir la segunda entrega

Primero activar el entorno e instalar `requirements.txt`. Si todavía no existen los tres CSV preparados, ejecutar una de las opciones del pipeline anterior. Se conserva la misma partición temporal y las mismas etiquetas de la primera entrega.

### Qué hace cada notebook

| Notebook | Contenido | ¿Entrena modelos? |
|---|---|---|
| `Preprocesamiento/06_preparación_secuencias.ipynb` | Reconstruye los canales horarios, normaliza y comprueba ventanas de 24, 72 y 168 horas. | No. |
| `Modelación/07_arquitecturas_secuenciales.ipynb` | Explica y dibuja LSTM, TCN y CNN-LSTM; comprueba sus formas, causalidad y gradientes. | No actualiza los pesos. |
| `Modelación/08_desbalance_y_entrenamiento.ipynb` | Compara pérdida ponderada y focal, longitudes de ventana y calibración descriptiva; muestra las curvas de aprendizaje. | Sí: hasta ocho ensayos de seis épocas. |
| `Modelación/09_comparación_y_anticipación.ipynb` | Compara con la línea base y calcula F1 por componente, cobertura de eventos y anticipación de alertas. | Reutiliza las redes disponibles o reconstruye las tres seleccionadas; vuelve a ajustar la línea base. |

Cada notebook se ejecuta con el kernel del entorno `reto-nov`, desde la raíz del proyecto o desde su carpeta. Ejecutar sus celdas en orden. Para revisar resultados ya guardados basta con abrir el notebook; no es necesario volver a entrenar.

Los notebooks 06 y 07 explican y comprueban las funciones compartidas. **No es necesario ejecutarlos antes del 08 o del 09**, porque estos importan los módulos de preparación y arquitecturas. El 09 también usa las celdas originales del 05 para reproducir la línea base, por lo que ese archivo debe conservarse en la ubicación indicada; no hace falta abrirlo ni ejecutarlo previamente.

### Ventanas en memoria y módulos compartidos

Los CSV preparados aportan las claves y las etiquetas. `Preprocesamiento/preparar_secuencias.py` reconstruye la historia desde los originales: cuatro sensores, cinco indicadores horarios de error y cuatro tiempos desde reemplazo. Son **13 canales por hora**, más modelo de máquina y antigüedad como entradas estáticas. Las medias y desviaciones de la línea base siguen en sus CSV; no se sobrescriben ni se necesitan para recuperar las mediciones originales.

Los ocho canales continuos se imputan, si fuera necesario, y se estandarizan con parámetros aprendidos solo de las 580,200 claves de entrenamiento. Los indicadores de error permanecen en 0/1. La antigüedad y el vocabulario del modelo también se ajustan solo con entrenamiento. No se recalculan estadísticas con validación.

Las ventanas se entregan por lotes con forma `B × L × 13`, sin exportar otro CSV ni duplicar todas las secuencias en un arreglo gigante. Las horas iniciales sin historia completa se excluyen. El entrenamiento del 08 usa **563,500 anclas comunes** para comparar longitudes con los mismos ejemplos; se mantienen las **144,000 horas de validación**. La línea base conserva sus 580,200 filas de entrenamiento. Esta diferencia se documenta al compararlas.

Para comprobar únicamente la preparación, desde la raíz del repositorio:

```bash
python Preprocesamiento/preparar_secuencias.py
```

El comando imprime el resumen y termina; no entrena ni deja datos disponibles en otra sesión. Para modelar, los notebooks importan y llaman `preparar_secuencias(ruta_repo)`.

Los otros módulos son:

- `Modelación/arquitecturas_secuenciales.py`: define las tres redes, las entradas estáticas y las dos salidas.
- `Modelación/entrenamiento_secuencial.py`: contiene las pérdidas, el ajuste, early stopping y las métricas compartidas.

**Los `.py` complementan los notebooks.** Se importan para reutilizar funciones; no sustituyen las explicaciones ni las tablas y gráficas. Deben conservarse en sus carpetas y no hay que ejecutarlos por separado para entrenar.

### Entrenamiento, reutilización y duración

El 08 fija semilla 42, CPU con un hilo, lotes de 512, AdamW con tasa 0.001, dropout 0.10, weight decay 0.0001 y recorte de gradiente a 1.0. Combina las pérdidas binaria y multiclase con pesos 0.60 y 0.40. Usa early stopping por PR-AUC de validación con paciencia 2, mejora de referencia 0.0001 y un máximo de seis épocas; recupera la mejor época observada.

El presupuesto declarado es de ocho ensayos: tres arquitecturas por dos pérdidas a 24 horas, más 72 y 168 horas para la combinación seleccionada. La búsqueda es manual y por etapas; no usa Optuna ni explora todos los hiperparámetros. En la ejecución entregada todos los ensayos llegaron a seis épocas. Esto no demuestra convergencia.

Los modelos y sus predicciones permanecen en la memoria de Python. En el 09:

- Si las variables `preparacion` y `resultados` del 08 siguen disponibles en el mismo kernel, se reutilizan los resultados de las redes seleccionadas. Abrir otro notebook no garantiza compartir esas variables.
- Si faltan, se reconstruye la preparación y se entrenan solo LSTM con focal, TCN con ponderada y CNN-LSTM con ponderada, todas a 24 horas y con la misma configuración original. No se repite la búsqueda de ocho ensayos.
- La regresión logística y XGBoost se vuelven a ajustar con el código y parámetros del 05 para recuperar sus predicciones y alinearlas con las mismas horas de validación.

La duración depende del equipo. Volver a ejecutar la celda de entrenamiento del 08 repite sus ajustes desde cero. Cerrar o reiniciar el kernel elimina los objetos en memoria; guardar el notebook conserva sus salidas visibles, pero **no guarda los pesos ni las matrices de probabilidades como objetos recuperables**. Las tablas, curvas y diagramas permanecen dentro de los notebooks, sin archivos permanentes de métricas, figuras o modelos de esta fase.

### Cómo se reproducen las métricas y alertas

El 09 calcula AP (la PR-AUC utilizada), ROC-AUC, recall con precisión horaria de al menos 0.80, F1 de cada componente y promedios de cinco clases y de los cuatro componentes. Cada umbral maximiza recall entre los puntos de validación que cumplen esa precisión. No se selecciona con prueba.

Una alerta se confirma en la tercera hora consecutiva sobre el umbral, usando la comparación `probabilidad >= umbral`. Los huecos o una hora por debajo interrumpen la racha. Cada racha emite una sola confirmación; esta se asocia a la siguiente falla de la máquina solo si ocurre después y dentro de 24 horas. Los componentes simultáneos cuentan como un evento máquina-hora para esta evaluación binaria.

Se exigen las 24 horas previas completas dentro de las anclas de validación. La anticipación se mide desde la confirmación y se resume entre eventos con alerta válida, junto con la cobertura. El diagnóstico distingue eventos sin confirmación dentro del horizonte y rachas ya confirmadas antes que permanecen activas: esas rachas no se vuelven a emitir ni se acreditan retrospectivamente. Una precisión de 0.80 por hora no garantiza la misma precisión por episodio.

La comparación de calibración del 08 es descriptiva; no se ajustó todavía un calibrador. Toda esta evaluación usa validación y corresponde al desarrollo del modelo. **Prueba permanece reservada para la tercera entrega.**

## Comprobaciones y reproducibilidad

El notebook 04 contrasta ejemplos de entrenamiento y validación con cálculos directos desde las fuentes: ventanas de sensores, conteos de errores, reemplazos y etiquetas. También verifica las claves, las fronteras y la estructura de los CSV exportados. El script incluye comprobaciones de integridad y separación temporal.

La preparación actual no usa muestreo ni operaciones aleatorias: con los mismos CSV, dependencias y código se esperan los mismos conjuntos. El notebook 05 fija la semilla 42 y los parámetros de los modelos. Al ejecutar sus celdas se regeneran las métricas de validación, las curvas y las matrices multiclase dentro del propio notebook, sin exportaciones adicionales. Estos son resultados de desarrollo; aún no hay evaluación final en prueba.

Los notebooks 06 y 07 comprueban además la preparación temporal, las formas de los lotes y la causalidad de las redes. El 09 verifica la alineación de claves y etiquetas y prueba la regla de alertas con ejemplos conocidos, incluyendo huecos y confirmación en la tercera hora. Las semillas y operaciones deterministas ayudan a repetir la ejecución; pueden existir pequeñas diferencias numéricas entre plataformas.
