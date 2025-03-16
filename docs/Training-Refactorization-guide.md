# Plan Detallado de Refactorización del Pipeline de Entrenamiento ✨✨✨

A continuación se presenta el plan de refactorización final con una estructura más clara, numeraciones coherentes y una revisión general de gramática y estilo.

---

## 1. Folder Structure Setup ✨✨✨

- Generar automáticamente la estructura requerida por Ultralytics:

```
dataset/
├── images
│   ├── train
│   ├── val
│   └── test
└── labels
    ├── train
    ├── val
    └── test
```

- Crear un archivo YAML de manera dinámica según el modo seleccionado (single/multi-class), con:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: <number_of_classes>
names: ["class1", "class2", ..., "classN"]
```

- Tras hacer el split, se elimina la carpeta descomprimida original para mantener el entorno limpio.
- Después de la fase de augment, se eliminan los datos de entrenamiento originales para evitar duplicaciones.
- Al finalizar, se borran archivos intermedios de la ejecución, conservando únicamente logs, modelos y métricas esenciales.

## 2. Data Augmentation ✨✨✨

- Aplicar Albumentations sobre las imágenes de entrenamiento.
- Definir cuántas variaciones por imagen (p. ej., 2) y limpiar ficheros innecesarios una vez finalizado.

## 3. Model Training ✨✨✨

- Seleccionar el modelo YOLOv8 (base u opción pre-entrenada).
- Ajustar la configuración de entrenamiento:
  - Modo single-class (`single_cls=True`) con nombres de clase personalizados.
  - Modo multi-class con parámetros estándar de YOLO.

## 4. Results Handling ✨✨✨

- Entrenar los modelos usando `epochs` y `batch_size` determinados.
- Archivar (en formato zip) los resultados en `repo/results` con un nombre basado en `{mode}_{datetime}.zip`.

---

## 5. Modular Structure (Optimized) ✨✨✨

A continuación se muestra el plan refactorizado con documentación a nivel de función:

```bash
train/
├── train_conf.py
│   └── class TrainingConfig (setup logging, hardware detection)
├── data_preprocessing.py
│   ├── unzip_dataset()
│   ├── validate_dataset_structure()
│   ├── create_yolo_structure()
│   ├── split_dataset()
│   └── augment_data()
├── train_utils.py
│   ├── initialize_model()
│   └── train_model()
└── results_handler.py
    ├── initialize_session_folder()
    ├── export_session_results()
    └── cleanup_old_sessions()
```

### `train_conf.py` ✨✨✨

**Objetivo**: Configuración global y logging.

- **`class TrainingConfig`**
  - **Misión**: Configurar logging, detectar hardware, definir hiperparámetros de entrenamiento.
  - **Método clave**:
    - `initialize()`: Organiza las rutas, logging y la detección de dispositivo.

### `data_preprocessing.py` ✨✨✨

**Objetivo**: Preparación integral del dataset—unzip, validación, estructura YOLO, split, augment.

- **`unzip_dataset(mode: str, force: bool = False) -> Path`**

  - **Meta**: Extraer el `.zip` de dataset según `mode`. Respeta `force` para re-extraer opcionalmente.

- **`validate_dataset_structure(mode: str, dataset_path: Path) -> (Path, Path)`**

  - **Meta**: Asegurar paridad imagen/etiqueta y estandarizar nombres (`images` y `labels`).

- **`create_yolo_structure(mode: str) -> Path`**

  - **Meta**: Crear subcarpetas YOLO (`images/{train,val,test}` y `labels/{train,val,test}`) en `cache/` y devolver su ruta.

- **`split_dataset(mode: str, images_path: Path, labels_path: Path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1) -> Path`**

  - **Meta**: Particionar datos en train/val/test. Tras el split, borrar los datos descomprimidos.

- **`augment_data(dataset_path: Path, augmentation_count=2) -> None`**

  - **Meta**: Aplicar Albumentations en imágenes de entrenamiento y limpiar datos no requeridos.

### `train_utils.py` ✨✨✨

**Objetivo**: Operaciones básicas de entrenamiento: inicializar modelo y llamada de YOLO.

- **`initialize_model(mode: str, single_cls: bool, class_names: list = None) -> YOLO`**

  - **Meta**: Instanciar el modelo YOLO, ya sea single o multi-class, con pesos base o pre-entrenados.

- **`train_model(model: YOLO, dataset_yaml: Path, epochs: int, batch_size: int, device: str, results_dir: Path) -> Path`**

  - **Meta**: Ejecutar el entrenamiento vía API Python (o CLI en fallback). Guardar logs, pesos, etc. y retornar la carpeta final.

### `results_handler.py` ✨✨✨

**Objetivo**: Gestionar sesiones de entrenamiento, archivado y limpieza.

- **`initialize_session_folder(results_base_dir: Path) -> Path`**

  - **Meta**: Crear `session_YYYYmmDD_HHMMSS` dentro de `results_base_dir`.

- **`export_session_results(session_dir: Path, archive_name: str) -> Path`**

  - **Meta**: Comprimir logs, pesos y métricas en `{archive_name}.zip`. Retorna la ruta final.

- **`cleanup_old_sessions(results_base_dir: Path, keep_last_n=5) -> None`**

  - **Meta**: Borrar sesiones antiguas, conservando las últimas `N`.

## 6. CLI Integration ✨✨✨

- Se integra con `lego_cli.py`.
- El usuario elige el `mode` (`bricks`, `studs`, `multiclass`) y, de ser necesario, el nombre de la clase para single-class.
- Se definen parámetros fijos por defecto para la demo, sin requerir configuraciones adicionales.

### 🎯 CLI Interface:

```bash
lego_cli.py training [COMMAND] [OPTIONS]...
```

### 🔑 Commands & Options:

#### 🌟 Global Options:

- `--debug`: Habilita logging detallado.

#### 🌟 One-Click Training:

```bash
lego_cli.py training one-click --mode [bricks|studs|multiclass] --epochs INT --batch-size INT
```

- Siempre ejecuta augment
- Detecta hardware (usa todos los GPUs disponibles)
- Genera zip final en `results/{mode}_{timestamp}.zip`
- Ejecuta limpieza final automáticamente

#### 📦 Unzip:

- `--mode [bricks|studs|multiclass]` *(requerido)*
- `--force/--no-force`: Re-extrae incluso si el dataset existe
- Limpia datos previos excepto el ZIP original en `presentation/`

#### ✅ Validate:

- `--mode [bricks|studs|multiclass]` *(requerido)*
- Estandariza nombres de subcarpetas `images` y `labels`
- Registra estadísticas detalladas con emojis

#### 🔄 Split:

- `--mode [bricks|studs|multiclass]` *(requerido)*
- `--train-ratio`, `--val-ratio`, `--test-ratio` *(por defecto: 0.7, 0.2, 0.1)*
- Crea estructura YOLO en la carpeta `cache`
- Limpia archivos intermedios

#### 🛠️ Augment:

- `--mode [bricks|studs|multiclass]` *(requerido)*
- `--augmentation-count INT`: número de aumentos por imagen (por defecto: 2)
- Muestra un resumen de la estadística de augment

#### 🏋️‍♂️ Train:

- `--mode [bricks|studs|multiclass]` *(requerido)*
- `--epochs INT` (por defecto: 50)
- `--batch-size INT` (por defecto: 16)
- Selecciona automáticamente el modelo pre-entrenado según el modo

#### 🧹 Cleanup:

- Elimina datos intermedios (`cache/`, split, ficheros de augment)
- No toca el ZIP original

---

## 7. Execution Examples ✨✨✨

### 🎯 One-click training (pipeline completo):

```bash
lego_cli.py training one-click --mode bricks --epochs 50 --batch-size 16
```

### 📌 Step-by-step approach:

1. Unzip:

```bash
lego_cli.py training unzip --mode multiclass
```

2. Validate:

```bash
lego_cli.py training validate --mode multiclass
```

3. Split:

```bash
lego_cli.py training split --mode multiclass --val-ratio 0.2
```

4. Augment:

```bash
lego_cli.py training augment --mode multiclass --augmentation-count 2
```

5. Train:

```bash
lego_cli.py training train --mode multiclass --epochs 100 --batch-size 32
```

##

