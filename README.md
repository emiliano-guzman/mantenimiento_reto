# Reto de mantenimiento predictivo

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

## Estado del proyecto

Por ahora solo se configura el entorno. El pipeline de datos, las semillas de los experimentos y los comandos para regenerar resultados se documentarán al implementarlos.
