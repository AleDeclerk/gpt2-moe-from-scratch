# GPT-2 con Mixture of Experts, desde cero

Un curso con forma de repositorio. Vos escribís el código, y los tests te dicen
si el código está bien.

El curso construye dos modelos de lenguaje. El primero es GPT-2, la
arquitectura densa de 2019. El segundo reemplaza cada bloque feed-forward por
una capa Mixture of Experts. Mixtral, DeepSeek y casi todos los modelos grandes
de hoy usan esa arquitectura. Al final comparás los dos, con la misma cantidad
de parámetros activos.

Acá no hay ninguna caja negra. Vos escribís el tokenizador, el attention, el
bloque del transformer, el loop de entrenamiento, el router y el loss de
balanceo de carga.

## Cómo funciona el curso

Cada capítulo es un directorio dentro de `chapters/`, con cuatro archivos:

| Archivo | Qué es |
|---|---|
| `README.md` | La teoría y la descripción de tu tarea |
| `exercise.py` | Tu código. Cada función arranca con `raise NotImplementedError` |
| `test_exercise.py` | Los tests. Verde quiere decir que tu código está bien |
| `solution.py` | El código de referencia, para el momento en que te trabás |

El paquete `gpt2moe/` lo construís vos. Ahora está casi vacío. Cuando pasan los
tests de un capítulo, promové tu código:

```bash
uv run python scripts/promote.py ch00
```

El comando copia tu `exercise.py` dentro de `gpt2moe/`, pero solo si los tests
pasan. El capítulo siguiente importa desde el paquete, así que importa tu
propio código. Para el capítulo 8 entrenás un modelo que es obra tuya, desde el
tokenizador hasta el optimizador.

## El sitio

<https://gpt2-moe-from-scratch.vercel.app>

El sitio muestra la teoría de cada capítulo y el estado del curso. Lee el
README de cada capítulo, así que el texto vive en un solo lugar.

El progreso es una medición, no una declaración:

1. Hacés pasar los tests de un capítulo.
2. Hacés push.
3. Una GitHub Action corre pytest para cada capítulo y escribe el resultado de
   cada test en `progress.json`.
4. Vercel vuelve a construir el sitio a partir de ese archivo.

El sitio no guarda estado propio, así que no puede mostrar un capítulo en verde
mientras los tests fallan. Para ver los números antes de un push, corré la
medición:

```bash
uv run python scripts/sync_progress.py
```

## Instalación

1. Instalá `uv`, si la computadora no lo tiene.
2. Corré `uv sync`. El comando crea el entorno e instala PyTorch.
3. Corré `uv run python scripts/get_data.py` para bajar el corpus.
4. Corré los primeros tests. Fallan, y ese es el arranque correcto.

```bash
uv run pytest chapters/ch00_tensors
```

## Comandos

| Comando | Qué hace |
|---|---|
| `uv run pytest chapters/ch00_tensors` | Testea un capítulo |
| `uv run pytest` | Testea todos los capítulos |
| `uv run pytest -m "not slow"` | Saltea los tests que necesitan una descarga |
| `MOE_TARGET=solution uv run pytest` | Testea el código de referencia, no el tuyo |
| `uv run python scripts/promote.py ch00` | Mueve tu código validado a `gpt2moe/` |
| `uv run python scripts/promote.py ch00 --from-solution` | Mueve el código de referencia, para seguir |

## Los capítulos

### Parte 1: la base

| Capítulo | Tema | Qué escribís |
|---|---|---|
| `ch00_tensors` | Tensores, softmax, gradientes | Un softmax estable, cross entropy, un backward hecho a mano |
| `ch01_data` | El pipeline de datos | Un tokenizador de caracteres, el split, el sampler de batches |
| `ch02_bpe` | Byte Pair Encoding | El algoritmo de BPE, y una comparación contra `tiktoken` |

### Parte 2: el transformer

| Capítulo | Tema | Qué escribís |
|---|---|---|
| `ch03_embeddings` | Embeddings y un baseline | Un modelo de bigramas, y el primer chequeo de cordura en `log(V)` |
| `ch04_attention` | Self-attention | Una cabeza, la máscara causal, la escala de `1/sqrt(d)` |
| `ch05_multihead` | Multi-head attention | La versión con loop, y la versión vectorizada |
| `ch06_block` | El bloque del transformer | La capa feed-forward, GELU, pre-LN, el camino residual |
| `ch07_gpt2` | El modelo completo | GPT-2, con weight tying y la inicialización original |
| `ch08_training` | El loop de entrenamiento | AdamW, warmup, cosine schedule, muestreo |

### Parte 3: Mixture of Experts

| Capítulo | Tema | Qué escribís |
|---|---|---|
| `ch09_moe` | La primera capa MoE | El router, la selección top-k, la combinación ponderada |
| `ch10_balance` | El problema del colapso | Métricas de uso de los expertos, el loss auxiliar, el z-loss del router |
| `ch11_capacity` | Capacidad y dispatch | El factor de capacidad, el descarte de tokens, un dispatch vectorizado |
| `ch12_compare` | Denso contra sparse | El experimento, y la comparación con la misma cantidad de parámetros activos |
| `ch13_ablations` | Ablaciones | Top-1 contra top-2, la cantidad de expertos, sin loss auxiliar |

## Cómo sabés que el modelo está bien

Cada parte termina con un test difícil de pasar por casualidad.

| Capítulo | Verde quiere decir |
|---|---|
| `ch05` | El attention vectorizado coincide con la versión que usa un loop |
| `ch07` | Tu GPT-2, con los pesos reales de `gpt2`, da los mismos logits que la referencia (atol 1e-4). Esto prueba que tu modelo **es** GPT-2 |
| `ch08` | El modelo denso baja del loss de validación objetivo |
| `ch11` | El MoE vectorizado coincide con el MoE naive |
| `ch12` | El modelo MoE le gana al modelo denso con la misma cantidad de parámetros activos |

## Hardware

Los modelos son chicos a propósito: 6 capas, 6 cabezas, 384 dimensiones, un
contexto de 256 tokens y alrededor de 10 millones de parámetros. Una corrida de
entrenamiento tarda minutos en la GPU de una máquina Apple Silicon, con el
backend MPS de PyTorch. En CPU también anda, con más paciencia.

Un test del capítulo 7 baja unos 500 MB de pesos desde Hugging Face. Lleva la
marca `slow`, así que el resto del curso funciona sin red.

## Créditos

El camino desde el modelo de bigramas hasta GPT-2 sigue la estructura de *Let's
build GPT* y `nanoGPT`, de Andrej Karpathy. El capítulo de BPE sigue `minbpe`,
del mismo autor. La parte de Mixture of Experts usa tres papers:

- Switch Transformer, Fedus et al., 2021
- ST-MoE, Zoph et al., 2022
- Mixtral of Experts, Jiang et al., 2024
