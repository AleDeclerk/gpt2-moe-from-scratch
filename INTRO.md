# Antes de empezar

## 1. Qué vas a construir

Un modelo de lenguaje, dos veces.

La primera vez es GPT-2, la arquitectura densa de 2019. La versión del curso es
chica: 6 capas, 6 cabezas, 384 dimensiones, contexto de 256 tokens, unos 10
millones de parámetros. Chica no quiere decir falsa: en el capítulo 7 cargás
adentro de tu código los pesos reales del GPT-2 de OpenAI, o sea los números
que ese modelo aprendió, y si tu salida coincide con la de referencia, lo que
escribiste es GPT-2.

La segunda vez reemplazás una sola pieza: la capa feed-forward de cada bloque
pasa a ser una capa Mixture of Experts de 8 expertos, de los que cada token usa
2. En el capítulo 12 comparás los dos modelos con los mismos parámetros
activos.

Lo del medio lo escribís vos: tokenizador, attention, bloque, loop de
entrenamiento, router y loss de balanceo de carga.

## 2. Qué es un modelo de lenguaje, de verdad

La tarea entra en una frase: **dado el texto hasta acá, dame una probabilidad
para cada token que puede venir después.**

Nada más. Si el vocabulario tiene 65 tokens, el modelo devuelve 65 números que
suman 1. Con el texto `El gato se subió al te`, un modelo entrenado le pone
probabilidad alta a `c` (por `techo`) y baja a `z`.

Lo que sale del modelo no son probabilidades todavía: son puntajes crudos que
se llaman **logits**, y un **softmax** los convierte en probabilidades. El
capítulo 0 lo explica y lo escribís ahí.

Generar texto es repetir esa única pregunta en un ciclo:

```
1. Le pasás el texto que tenés.
2. El modelo devuelve una probabilidad por token.
3. Sorteás un token con esas probabilidades (eso es el muestreo).
4. Lo pegás al final del texto y volvés al paso 1.
```

Todo lo demás del curso es maquinaria para esa única pregunta.

## 3. El vocabulario mínimo

**Token.** Un pedazo de texto que el modelo trata como unidad indivisible. El
tokenizador de caracteres del capítulo 1 parte `hola` en cuatro tokens; el BPE
del capítulo 2 junta los pares que más se repiten y puede darte `hola` en uno
solo.

**Vocabulario.** La lista cerrada de tokens que existen para ese modelo, y el
modelo no puede emitir nada de afuera. El corpus del curso tiene 65 caracteres
distintos, así que su vocabulario es de 65 tokens. El de GPT-2 tiene 50257.

**Embedding.** Un token es un entero, y un entero no dice nada sobre parecidos:
`3` no está más cerca de `4` que de `61`. Un embedding es un vector de números
por cada token, por ejemplo 384, guardados en una tabla de 65 filas por 384
columnas. Entrenando, los tokens que aparecen en contextos parecidos terminan
con vectores parecidos. Esa tabla es un **tensor**: una grilla de números con
un `shape` declarado, acá `(65, 384)`.

**Parámetro.** Cualquier número del modelo que el entrenamiento ajusta. Los
65 × 384 = 24960 números de esa tabla son parámetros, y el modelo del curso
tiene cerca de 10 millones.

**Loss.** Un número que mide qué tan equivocada estuvo una predicción; más bajo
es mejor. El curso usa cross entropy: el logaritmo negativo de la probabilidad
que el modelo le dio al token correcto. Si le dio 0.9, el loss da 0.105; si le
dio 0.001, da 6.9.

**Gradiente.** Si muevo este parámetro un poquito, ¿el loss sube o baja, y
cuánto? Si le sumás 0.001 y el loss sube 0.002, la pendiente vale 2, así que
conviene bajarlo. Esa pendiente para un parámetro solo, con los demás quietos,
es una **derivada parcial**, y el gradiente es el paquete con las de todos
juntos. Medirlas de a una sobre 10 millones de números sería carísimo: la
**regla de la cadena** las saca todas de una sola pasada hacia atrás.

**Entrenamiento.** Pasás un batch de datos por el modelo (el **forward**),
calculás el loss, calculás los gradientes (el **backward**) y movés cada
parámetro un pasito en la dirección que baja el loss. Repetís miles de veces.

**Inferencia.** Usar el modelo ya entrenado: solo forward, sin gradientes y con
los parámetros quietos. Generar texto es inferencia.

## 4. Qué es un transformer

Es una pila de bloques iguales en forma, y cada bloque hace dos cosas. Primero
mezcla información entre posiciones: eso es el **self-attention**, donde cada
posición mira las anteriores, decide cuánto le importa cada una y arma un
promedio ponderado (una **máscara causal** le prohíbe mirar hacia adelante:
espiar el token siguiente haría trivial la tarea). Después procesa cada
posición por separado: eso es el feed-forward, la misma función chiquita
aplicada a cada posición sin mirar a las vecinas. Eso, seis veces. Al final una
capa de salida da un logit por cada token del vocabulario, y volvés a la
sección 2.

## 5. Qué es Mixture of Experts, y por qué existe

En el modelo denso cada token pasa por todos los parámetros, así que cada
parámetro nuevo se paga en cómputo por cada token.

Mixture of Experts rompe ese vínculo. En vez de un feed-forward que procesa
todos los tokens, hay varios, y cada uno es un experto. Un **router**, que es
una capa lineal chiquita, puntúa a los expertos para cada token, se queda con
los 2 mejores (eso es **top-k**, con k igual a 2) y combina sus salidas
ponderadas por esos puntajes. Los otros 6 no corren para ese token.

Así el modelo tiene unas 4 veces más parámetros con casi el mismo costo por
token, y esa es la razón por la que los modelos grandes de hoy son **sparse**:
cada token activa solo una parte de los parámetros, en vez de todos. La parte 3
te muestra la factura: un router librado a su suerte manda casi todos los
tokens al mismo experto, y evitarlo pide tres piezas que escribís vos. Un
**loss de balanceo de carga**, que es un castigo extra cuando el reparto entre
expertos queda desparejo. Un **z-loss**, que evita que los puntajes del router
se vayan a números enormes y rompan las cuentas. Y un **límite de capacidad**,
que es un tope de tokens por experto en cada batch.

## 6. Qué necesitás saber, y qué no

Necesitás programar bien en Python y ganas de escribir código que falla
primero, porque acá se arranca siempre con los tests en rojo.

No necesitás acordarte de álgebra lineal ni de cálculo: cada concepto se
explica cuando aparece, con un ejemplo numérico antes de la fórmula. Tampoco
hace falta PyTorch previo ni una GPU cara.

## 7. Cómo se trabaja

Cada capítulo es un directorio dentro de `chapters/` con cuatro archivos: un
`README.md` con la teoría y tu tarea, un `exercise.py` donde escribís el
código, un `test_exercise.py` que dice si está bien, y un `solution.py` de
referencia para cuando te trabás.

Leés el README, completás las funciones de `exercise.py` (todas arrancan con
`raise NotImplementedError`) y corrés los tests:

```bash
uv run pytest chapters/ch00_tensors
```

Cuando pasan, promovés tu código al paquete `gpt2moe/`:

```bash
uv run python scripts/promote.py ch00
```

`promote.py` copia tu `exercise.py` adentro del paquete, y solo si los tests
pasan. El capítulo siguiente importa desde `gpt2moe/`, o sea que importa código
tuyo: para el capítulo 8 entrenás un modelo que es tuyo de punta a punta.

El detalle (instalación, lista de capítulos, comandos, medición del progreso)
está en el README del repositorio.
