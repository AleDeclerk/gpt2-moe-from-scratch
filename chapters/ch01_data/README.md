# Capítulo 1: el pipeline de datos

## Qué aprende un modelo de lenguaje

La tarea entra en una sola frase: dado el texto hasta acá, asignarle una
probabilidad a cada token que puede venir después. Todo el resto del curso es
maquinaria para esa única pregunta.

Por eso los datos de entrenamiento no son una tabla de entradas y etiquetas.
Son una sola secuencia larga de tokens, y la etiqueta de cada posición es el
token de la posición siguiente. El dataset se escribe solo, y de ahí sale el
nombre del método: self-supervised.

## 1. Un tokenizador de caracteres

Un tokenizador mapea texto a enteros. La versión más simple usa un entero por
carácter. Tomás el conjunto de caracteres del corpus, lo ordenás, y usás la
posición en esa lista ordenada como identificador.

Tiny Shakespeare tiene 65 caracteres distintos, así que el vocabulario es de 65
tokens. Es un vocabulario muy chico. GPT-2 usa 50257 tokens, y el capítulo 2
construye un tokenizador de ese tipo con el algoritmo BPE.

Los dos tamaños se compensan entre sí:

| | Tokens de caracteres | Tokens BPE |
|---|---|---|
| Vocabulario | 65 | 50257 |
| Tokens para el mismo texto | muchos | unas 4 veces menos |
| Tabla de embedding | chica | grande |
| Un contexto de 256 tokens abarca | unos 256 caracteres | unos 1000 caracteres |

Un vocabulario chico da secuencias largas, y el costo de attention crece con el
cuadrado del largo de la secuencia. Un vocabulario grande da secuencias cortas,
pero también una tabla de embedding grande y una capa de salida grande.

**El orden importa.** Dos corridas tienen que darle el mismo identificador al
mismo carácter, porque un modelo guardado guarda los identificadores, no los
caracteres. `sorted(set(text))` es estable. `set(text)` solo, no.

## 2. El split de entrenamiento y validación

El curso reserva el último 10 por ciento del texto para validación, y el split
es contiguo. Un split aleatorio de posiciones acá está mal, y la razón es propia
de los datos secuenciales.

Los ejemplos de entrenamiento son ventanas sobre el texto, y las ventanas se
superponen. Un split aleatorio manda la ventana de la posición 100 al conjunto
de entrenamiento, y la ventana de la posición 101 al conjunto de validación. Las
dos ventanas comparten 255 de sus 256 caracteres. Así el loss de validación mide
memoria, no generalización, y se ve mucho mejor de lo que realmente es.

## 3. Un batch, muchos ejemplos

Esta es la parte que sorprende. Un bloque de 256 tokens no es un ejemplo de
entrenamiento. Son 256 ejemplos de entrenamiento en un solo tensor.

Tomá el bloque `[F, i, r, s, t]`. El modelo ve esto:

```
input                 target
[F]                -> i
[F, i]             -> r
[F, i, r]          -> s
[F, i, r, s]       -> t
```

La máscara causal del capítulo 4 hace las cuatro predicciones en un solo
forward. La posición `t` puede leer las posiciones de `0` hasta `t`, y nada
después. Así que el código no construye las cuatro filas de arriba. Construye
dos tensores:

```
x = data[i     : i + T]        el bloque
y = data[i + 1 : i + T + 1]    el mismo bloque, corrido una posición
```

Entonces `y[t]` es la respuesta correcta para el prefijo que termina en `x[t]`.
Un tensor de shape `(B, T)` contiene `B * T` predicciones. Con `B = 32` y
`T = 256`, un paso de entrenamiento usa 8192 ejemplos.

**Lo que implica para el offset.** La posición inicial `i` tiene que cumplir
`i + T + 1 <= len(data)`, porque `y` lee una posición más allá que `x`. Un
offset sacado de `len(data) - T` se pasa por uno y falla en el último bloque.

## Tu tarea

1. Abrí `exercise.py`.
2. Escribí `CharTokenizer`, `train_val_split` y `get_batch`.
3. Corré los tests.

```bash
uv run pytest chapters/ch01_data
```

4. Promové el código.

```bash
uv run python scripts/promote.py ch01
```

Los tests usan un texto chico propio, así que no necesitan bajar nada. Para
conseguir el corpus real del capítulo 8, corré `uv run python scripts/get_data.py`.

## Preguntas para responderte a vos mismo

- Tiny Shakespeare tiene 1115394 caracteres. ¿Cuántos tokens necesita el
  tokenizador BPE de GPT-2 para ese mismo texto? El capítulo 2 lo mide.
- Con `B = 32` y `T = 256`, ¿cuántos bytes ocupa un batch en `int64`? ¿Y en
  `int32`?
- El split de validación es el final del texto. Imprimí los últimos 500
  caracteres. ¿Esa parte del corpus es el mismo tipo de texto que el comienzo?
