# Capítulo 2: Byte Pair Encoding

## Por qué no alcanza con los caracteres

El tokenizador del capítulo 1 tiene dos problemas, y los dos salen de la misma
decisión: armar el vocabulario con los caracteres de un solo corpus.

El primer problema es la cobertura. Un carácter que no aparece en Tiny
Shakespeare no tiene identificador, así que `encode` tira un KeyError. La letra
`ñ`, un emoji y un carácter chino fallan igual.

El segundo problema es el largo. Un carácter da un token, así que el modelo
necesita un contexto largo para un texto corto. El costo de attention crece con
el cuadrado del largo de la secuencia, o sea que el largo se paga caro.

BPE resuelve los dos. GPT-2 lo usa, y su vocabulario tiene 50257 tokens.

## 1. Los bytes como vocabulario base

BPE no arranca de los caracteres. Arranca de los bytes.

Todo texto tiene una representación en UTF-8, y todo byte UTF-8 es un número
entre 0 y 255. Con eso, un vocabulario base de 256 tokens alcanza para
representar cualquier texto en cualquier idioma, sin token desconocido. Esa
propiedad es la razón de la elección.

```python
"hola".encode("utf-8")   # b'hola'        -> [104, 111, 108, 97]
"ñ".encode("utf-8")      # b'\xc3\xb1'    -> [195, 177]
```

La segunda línea muestra el costo: un carácter se convirtió en dos tokens. BPE
recupera ese costo con los merges.

## 2. El algoritmo de entrenamiento

El algoritmo tiene tres pasos y los repite:

1. Contá cuántas veces aparece cada par de tokens adyacentes.
2. Agarrá el par más frecuente y dale un identificador nuevo.
3. Reemplazá cada aparición de ese par por el identificador nuevo.

Un ejemplo con la palabra `aaabdaaabac`, en bytes:

```
inicio          a a a b d a a a b a c
par más frecuente: (a, a), 4 veces.  Z = aa
tras el merge   Z a b d Z a b a c
par más frecuente: (Z, a), 2 veces.  Y = Za
tras el merge   Y b d Y b a c
par más frecuente: (Y, b), 2 veces.  X = Yb
tras el merge   X d X a c
```

Los once tokens se volvieron cinco. El vocabulario pasó de 4 a 7. Ese es el
trade-off de BPE, y el argumento `vocab_size` lo controla.

**Ojo con el solapamiento.** En `[1, 1, 1]` el par `(1, 1)` aparece dos veces,
pero un merge puede reemplazar solo uno. El resultado es `[Z, 1]`, no `[Z, Z]`
ni `[Z]`. Un loop con un índice explícito lo maneja bien. Una llamada a
`list.replace`, no.

## 3. encode necesita el orden de los merges

Los merges no son un conjunto. Son una lista ordenada, y `encode` tiene que
aplicarlos en el mismo orden que `train`.

La razón es que los merges de después se construyen sobre los de antes. El
token `X` del ejemplo de arriba significa `aaab`, pero solo porque `Y` y `Z` ya
existen. Un `encode` que aplique `X` primero da otro resultado, y el modelo
nunca vio esos identificadores durante el entrenamiento.

Entonces `encode` repite un solo paso: de todos los pares que aparecen en la
lista actual, buscá el de menor índice de merge y aplicalo. Frená cuando ningún
par de la lista esté en la tabla de merges.

## 4. Lo que decode tiene que aguantar

`decode` concatena los bytes de cada token y después decodifica UTF-8. Una
lista cualquiera de identificadores puede dar una secuencia de bytes que no sea
UTF-8 válido, porque un carácter de varios bytes puede quedar cortado por la
mitad. Durante la generación, el modelo puede producir justo eso.

La respuesta estándar es `errors="replace"`, que pone el carácter `U+FFFD`
donde estaban los bytes inválidos. La alternativa, una excepción, corta la
generación del modelo por un motivo que no tiene ningún interés.

## Tu tarea

1. Abrí `exercise.py`.
2. Escribí `get_stats`, `merge` y los tres métodos de `BPETokenizer`.
3. Corré los tests.

```bash
uv run pytest chapters/ch02_bpe
```

4. Promové el código.

```bash
uv run python scripts/promote.py ch02
```

Las dos funciones van primero, porque `train` y `encode` las usan. Ponelas en
verde antes de arrancar con la clase.

## Preguntas para responderte a vos mismo

- Entrená el tokenizador con Tiny Shakespeare y `vocab_size = 512`, y medí la
  compresión. El test `test_compression_ratio` imprime el número.
- El tokenizador real de GPT-2 corta el texto con una expresión regular antes
  de BPE, así que un merge nunca cruza el límite de una palabra. ¿Qué se rompe
  sin ese paso? Fijate en qué se convierten `" the"`, `" the."` y `" the,"`.
- El vocabulario de GPT-2 tiene 50257 tokens: son 256 bytes, más 50000 merges,
  más uno. ¿Cuál es ese último, y para qué lo necesita un modelo?
