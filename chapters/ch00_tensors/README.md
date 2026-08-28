# Capítulo 0: tensores, softmax y gradientes

## Por qué este capítulo va primero

Hay tres operaciones que aparecen en todos los capítulos que siguen. Un
softmax sobre la última dimensión convierte los scores de attention en pesos,
y también convierte los logits finales en una probabilidad para cada token. Un
loss de cross entropy mide qué tan equivocadas están esas probabilidades. La
regla de la cadena lleva ese error de vuelta hasta cada parámetro.

PyTorch tiene las tres, y de acá en adelante el curso usa la versión de
PyTorch. Las escribís una sola vez, acá, porque un bug en un capítulo
posterior se encuentra mucho más rápido cuando ya sabés qué hace cada una de
estas operaciones.

## 1. Un softmax estable

La definición es simple. Para una fila de números `z`:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Pasar esa fórmula directo a código falla. Con `z_i = 1000`, `exp(1000)` es más
grande que el float más grande que existe, así que el resultado es `inf`.
Después `inf / inf` da `nan`, y ese `nan` se propaga por todo el modelo.

La corrección usa una propiedad del softmax: si le restás una constante `c` a
cada elemento de `z`, el resultado no cambia:

```
exp(z_i - c) / sum_j exp(z_j - c) = [exp(z_i) exp(-c)] / [exp(-c) sum_j exp(z_j)]
```

El factor `exp(-c)` está en el numerador y en el denominador, así que se
cancela. La elección útil es `c = max(z)`. Después de la resta el elemento más
grande queda en `0`, así que `exp` devuelve un número entre `0` y `1`. No hay
overflow posible.

Esto no es un detalle de estilo. GPT-2 divide los scores de attention por la
raíz cuadrada de la dimensión de la cabeza por una razón parecida, y el
capítulo 4 la explica.

**Shapes.** Tu función recibe un tensor 2-D con shape `(N, C)` y devuelve un
tensor con el mismo shape. Cada fila tiene que sumar `1`. Mirá bien el
argumento `keepdim` de `max` y de `sum`: una reducción sin `keepdim` saca la
dimensión y rompe el broadcast.

## 2. Cross entropy

El modelo le da un score, que se llama logit, a cada uno de los `C` tokens del
vocabulario. El token correcto que sigue tiene índice `t`. El loss es el
logaritmo negativo de la probabilidad que el modelo le da a `t`:

```
loss = -log(softmax(z)_t)
```

Un modelo perfecto le da probabilidad `1` al token correcto, y `-log(1)` es
`0`. Un modelo que le da probabilidad `0.001` se lleva un loss de más o menos
`6.9`.

Acá también hay una trampa. La expresión `log(softmax(z))` calcula la
exponencial y después el logaritmo, y esas dos operaciones juntas pierden
precisión. El álgebra te saca ese ida y vuelta de encima:

```
log_softmax(z)_i = z_i - max(z) - log(sum_j exp(z_j - max(z)))
```

Tu función toma logits con shape `(N, C)` y targets con shape `(N,)`, y
devuelve un solo escalar: el loss promedio sobre las `N` filas.

**Un número para acordarse.** Un modelo sin entrenar le da más o menos la
misma probabilidad a cada token, así que el loss ronda `log(C)`. Con un
vocabulario de 65 caracteres ese valor da cerca de `4.17`. El capítulo 3 usa
este número como primer test del primer modelo. Un entrenamiento que arranca
muy por encima de `log(C)` tiene un bug en la inicialización.

## 3. El backward de una capa lineal

Una capa lineal es `y = x @ W + b`. Durante el entrenamiento, PyTorch te da el
gradiente del loss respecto de la salida, `dL/dy`. Te faltan tres gradientes
más.

La regla para un producto de matrices es corta:

```
dL/dx = dL/dy @ W.T
dL/dW = x.T @ dL/dy
dL/db = dL/dy sumado sobre la dimensión del batch
```

Con mirar los shapes alcanza para acordarse. Con `x` de shape `(N, in)`, `W`
de shape `(in, out)` y `dL/dy` de shape `(N, out)`, hay un solo orden de cada
producto que da el shape correcto. El gradiente del bias es una suma: el mismo
`b` se suma a las `N` filas, así que las `N` filas le mandan error.

Tu función devuelve los tres gradientes como una tupla. El test los compara
contra `loss.backward()` de PyTorch. Si los números dan iguales, tu derivación
está bien.

## Tu tarea

1. Abrí `exercise.py`.
2. Escribí las tres funciones. No uses `torch.softmax`, `torch.log_softmax`,
   `torch.nn.functional.cross_entropy` ni `backward`.
3. Corré los tests.

```bash
uv run pytest chapters/ch00_tensors
```

4. Cuando los tests pasen, promové el código.

```bash
uv run python scripts/promote.py ch00
```

## Preguntas para responderte a vos mismo

- El test `test_softmax_is_stable` usa logits de `1000`. Sacá la resta del
  máximo y correlo. ¿Qué te da?
- ¿Por qué `dL/db` necesita una suma y `dL/dW` necesita un producto de
  matrices?
- El loss promedio divide por `N`. ¿Qué gradiente cambia si el loss usa una
  suma en vez del promedio, y en cuánto?
