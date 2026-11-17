# Reto de mantenimiento predictivo

Preparación de datos para anticipar si una máquina tendrá una falla en las siguientes 24 horas y qué componente fallará. Se utilizan los cinco archivos del conjunto Microsoft Azure Predictive Maintenance.

## Estado del proyecto

El proyecto incluye preparación de datos, línea base clásica, modelos secuenciales y una tercera etapa con Transformer, incertidumbre, calibración, interpretabilidad y política de mantenimiento. La evaluación final de prueba está guardada en el notebook 13: XGBoost con umbral 0.945, elegido con validación antes de esa evaluación. Los notebooks conservan sus explicaciones, tablas y gráficas; abrirlos permite consultar los resultados sin volver a entrenar.

### Cierre de la primera entrega

La primera fase queda identificada con el tag **`entrega1`**, que conserva la versión del código correspondiente al reporte: revisión y EDA, análisis de mantenimiento, variables históricas, etiquetas, partición temporal y comparación de regresión logística y XGBoost en validación. Los resultados están en `Modelación/05_línea_base_clásica.ipynb`; el reporte se entrega por separado.

La versión entregada puede consultarse en [GitHub: entrega1](https://github.com/emiliano-guzman/mantenimiento_reto/tree/entrega1). El trabajo de las siguientes fases podrá continuar en `main` sin mover este tag. En esa versión, prueba todavía no se había evaluado.

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

Para volver a la rama de trabajo: `git switch main`. Guardar o resguardar los cambios locales antes de cambiar de versión. En esa segunda versión, prueba seguía reservada para la tercera etapa, con el modelo y la política por elegir en validación.


### Tercera entrega

Los notebooks 10–13 añaden el Transformer, MC Dropout y temperature scaling, Integrated Gradients por componente y una política económica comparada con operar hasta la falla y reemplazar por calendario. Se mantiene la partición temporal original. El modelo final es XGBoost; los análisis de calibración e interpretabilidad del Transformer describen esa red y no se trasladan a XGBoost.

Las versiones anteriores permanecen identificadas por `entrega1` y `entrega2`. La sección de reproducción de esta tercera etapa explica qué se ejecuta, qué se reutiliza y cómo consultar la evaluación final guardada.


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
├── 10_transformer.ipynb
├── 11_incertidumbre_calibración.ipynb
├── 12_interpretabilidad.ipynb
├── 13_política_mantenimiento.ipynb
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

Este aislamiento ya está incluido: no hay que ejecutar un programa adicional ni conservar el intercambio temporal. El notebook 13 también aísla XGBoost y mantiene temporalmente sus pipelines ajustados para aplicar exactamente los mismos objetos a prueba después de fijar la política; elimina esa carpeta al terminar. El notebook 05 puede ejecutarse por separado en su propio kernel.

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

**Prueba se prepara y guarda desde el pipeline; su evaluación final se realiza en el notebook 13**, después de elegir modelo y umbral con validación. Los análisis de desbalance del notebook se limitan a entrenamiento y validación. No se utiliza prueba para seleccionar variables, modelos, hiperparámetros o umbrales.

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

La comparación de calibración del 08 es descriptiva; no se ajustó todavía un calibrador. Toda esta evaluación usa validación y corresponde al desarrollo del modelo. **Los notebooks de la segunda etapa no evalúan prueba; la evaluación final está en el 13.**

## Reproducir la tercera entrega

Instalar las dependencias y regenerar los CSV preparados como se explica arriba. Conservar los notebooks anteriores y los módulos `.py` en sus ubicaciones: se reutilizan sus definiciones y algunas tablas guardadas. Los notebooks 10–13 pueden abrirse desde la raíz o desde `Modelación/`. Ejecutar las celdas de cada uno en orden solo si se desea reproducir sus cálculos; para consultar los resultados entregados basta leer sus salidas.

| Notebook | Qué realiza | Reutilización y entrenamiento |
|---|---|---|
| `Modelación/10_transformer.ipynb` | Codificador Transformer con posición y token de máquina, dos salidas, entrenamiento y comparación en validación. | Ajusta una configuración con semilla 42; conserva la mejor época por AP. Lee la tabla guardada del 09 para comparar. |
| `Modelación/11_incertidumbre_calibración.ipynb` | Veinte pasadas MC Dropout, Brier, diagramas de confiabilidad y ajuste de temperatura para ambas salidas. | Reutiliza el Transformer disponible o reconstruye su ajuste original. Dropout y calibración no actualizan sus pesos. |
| `Modelación/12_interpretabilidad.ipynb` | Integrated Gradients: canales y horas de alertas por componente, aciertos y errores, sensibilidad a referencias y contraste con el EDA. | Reutiliza o reconstruye el Transformer. IG no entrena; requiere la tabla por componente guardada en el EDA 02. |
| `Modelación/13_política_mantenimiento.ipynb` | Selección económica en validación, sensibilidad, evaluación final en prueba y comparación con dos referencias. | Ajusta XGBoost con las definiciones del 05; reutiliza CNN-LSTM y Transformer o reconstruye solo sus ajustes originales. Solo el ganador pasa a prueba. |

### Dependencias entre notebooks y objetos en memoria

No es necesario abrir y ejecutar toda la secuencia 01–13 para cada análisis. Los módulos de preparación y entrenamiento se importan directamente. El 10 necesita las salidas comparativas del 09 y el 12 la tabla del EDA 02: esas salidas están incluidas en los notebooks versionados. No borrarlas si se desea usar estas referencias sin recalcularlas.

Los notebooks 11 y 12 reutilizan `resultado_transformer` y `preparacion_10` cuando existen en el mismo kernel. El 13 también puede reutilizar `resultados` del entrenamiento secuencial. Abrir un notebook distinto no garantiza compartir kernel. En una sesión nueva se reconstruyen los ajustes originales necesarios, por lo que la ejecución puede durar varios minutos y consumir memoria. No se exportan pesos permanentes: guardar un notebook conserva sus resultados visibles, no sus objetos de Python.

El Transformer usa ventanas de 24 horas y 13 canales, dimensión 32, cuatro cabezas y dos capas; conserva el protocolo de entrenamiento del proyecto: CPU con un hilo, semilla 42, lotes de 512, AdamW, dropout, weight decay y parada temprana por AP de validación. MC Dropout utiliza 20 pasadas con semillas explícitas, sin modificar parámetros ni buffers. IG empieza con 64 pasos y refina a 256 o 512 cuando hace falta para cumplir la tolerancia numérica declarada.

### Calibración e interpretabilidad sin consultar prueba

El 11 ajusta las temperaturas con septiembre, dejando 24 horas de margen antes de octubre, y comprueba después en octubre. Son 69 600 anclas de ajuste, 2 400 de margen y 72 000 de comprobación. Octubre ya había participado en el desarrollo del modelo: esta comprobación no constituye un conjunto independiente ni reemplaza a prueba. Una mejora de Brier o log loss no garantiza mejorar todas las medidas de calibración; las tablas registran también esas diferencias.

El 12 explica logits de la función determinista del Transformer, con dropout apagado. Usa referencias sanas de entrenamiento y casos de octubre; muestra contribuciones positivas y negativas por hora y canal. El contraste con el EDA compara sensores y direcciones, respetando que sus escalas y orígenes temporales son distintos. Las atribuciones describen el modelo respecto a referencias, no causas físicas. Modelo de máquina y antigüedad se mantienen fijos; no se les asigna una atribución propia.

### Política, costos y evaluación final

En el 13 se comparan XGBoost, CNN-LSTM ponderada con 24 horas y Transformer usando las mismas horas de validación. La cuadrícula contiene 1 002 alternativas por modelo: umbrales de 0 a 1 en pasos de 0.001 y no alertar. La decisión elegida fue **XGBoost con umbral 0.945**. Ese objetivo económico es distinto del umbral que exige precisión horaria de al menos 0.80 en las etapas anteriores.

Una racha de tres horas sobre el umbral emite una sola acción; el componente propuesto es el más probable entre comp1–comp4 en la confirmación. Una acción se acredita solo si la siguiente falla de ese componente ocurre estrictamente después y dentro de 24 horas. Se acredita una sola intervención por falla y las acciones adicionales no reciben otro beneficio. Aquí la unidad es falla de componente, mientras que la anticipación binaria del 09 agrupaba componentes simultáneos por máquina y hora.

Los costos son 15 000 por falla no anticipada, 3 000 por intervención acertada y 600 por acción sin un nuevo acierto. La sensibilidad varía la razón falla/falsa alarma entre 5, 10, 25, 50 y 100, manteniendo el acierto en 3 000 y la falsa alarma en 600; las alternativas se eligen únicamente con validación.

Se compara con operar hasta la falla y con un calendario por componente entre 7, 14, 30, 60 y 90 días. Al permitir no programar, validación eligió esa opción. El notebook documenta que la referencia activa se completó en revisión después de la primera inferencia final, sin cambiar modelo ni umbral y sin nuevas predicciones de prueba. Sus intervalos finitos se eligieron de los candidatos ya evaluados en validación: 90 días para cada componente. La tabla no proporciona el costo de sustituir un componente sano; usar 600 para esa acción de calendario es un supuesto explícito.

La evaluación final guardada sobre noviembre–diciembre contiene los siguientes costos, bajo esas reglas:

| Política | Costo total en prueba |
|---|---:|
| Predictiva: XGBoost, umbral 0.945 | 453 000 |
| Operar hasta la falla | 1 815 000 |
| Calendario activo por componente | 1 884 600 |

El 13 registra el acta y su huella antes de la inferencia final del modelo elegido. En el mismo kernel, repetir la celda final reutiliza los resultados si la huella coincide. **Para revisar prueba, leer las salidas ya guardadas.** En un kernel nuevo, ejecutar todo reconstruye el protocolo y reproduce sus resultados; no debe convertirse en una búsqueda de mejores decisiones mirando prueba. No se ajustan normalizadores, pesos, temperaturas ni umbrales con ese conjunto.

### Alcance de los resultados y datos reales

La exactitud global (`accuracy`) no se utiliza como métrica principal ante el desbalance. Se presentan AP/PR-AUC, ROC-AUC, precisión y recall, F1 por componente, calibración, anticipación y costos según la etapa. Las métricas por hora, por acción y por falla tienen denominadores diferentes y no se intercambian.

Los datos son simulados. En una planta real, la deriva de sensores podría exigir vigilar sus distribuciones y revisar las escalas; los faltantes podrían requerir reglas basadas en disponibilidad y evaluación temporal; los mantenimientos incompletos volverían inciertas las horas desde reemplazo. Contar ausencia de errores como cero supone que el registro está completo: con interrupciones de captura habría que distinguir ausencia de evento de ausencia de información. Estas situaciones no se simularon en este proyecto.

Los costos representan una reproducción retrospectiva: las acciones virtuales no cambian los sensores, las fallas ni los mantenimientos históricos posteriores. Tampoco se modelan repuestos, técnicos o duración de paros. La diferencia de costos no demuestra ahorro real ni fallas efectivamente evitadas. Se incluyen las fallas cercanas a fronteras, mostrando cuáles no tienen las 24 horas anteriores completas en las anclas disponibles.

La sección 6 del notebook 13 añade una revisión temporal conjunta de cuatro máquinas de validación: probabilidad binaria, confirmaciones de alerta, errores por tipo, reemplazos y fallas por componente. Selecciona ejemplos por fecha y componente, sin elegirlos por aciertos; reutiliza las probabilidades existentes y no entrena ni consulta prueba. Se añadió como diagnóstico descriptivo después de fijar el protocolo y no modificó decisiones finales. Las curvas agregadas del EDA y los mapas de IG complementan esta inspección, sin demostrar causalidad.

## Comprobaciones y reproducibilidad

El notebook 04 contrasta ejemplos de entrenamiento y validación con cálculos directos desde las fuentes: ventanas de sensores, conteos de errores, reemplazos y etiquetas. También verifica las claves, las fronteras y la estructura de los CSV exportados. El script incluye comprobaciones de integridad y separación temporal.

La preparación actual no usa muestreo ni operaciones aleatorias: con los mismos CSV, dependencias y código se esperan los mismos conjuntos. El notebook 05 fija la semilla 42 y los parámetros de los modelos. Al ejecutar sus celdas se regeneran las métricas de validación, las curvas y las matrices multiclase dentro del propio notebook, sin exportaciones adicionales. Los resultados del 05 son de desarrollo en validación; la evaluación final de prueba se encuentra en el 13.

Los notebooks 06 y 07 comprueban además la preparación temporal, las formas de los lotes y la causalidad de las redes. El 09 verifica la alineación de claves y etiquetas y prueba la regla de alertas con ejemplos conocidos, incluyendo huecos y confirmación en la tercera hora. Las semillas y operaciones deterministas ayudan a repetir la ejecución; pueden existir pequeñas diferencias numéricas entre plataformas.

Los notebooks 10–12 comprueban arquitectura, parámetros sin cambios durante MC/IG y precisión numérica de las atribuciones. El 13 verifica continuidad de alertas, duplicados, componente equivocado y horizonte estrictamente futuro. Las comprobaciones acompañan los resultados; no sustituyen una auditoría de la disponibilidad real de cada registro.
