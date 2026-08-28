# Capítulo 2: Byte Pair Encoding

## Por qué no alcanza con los caracteres

El tokenizador del capítulo 1 tiene dos problemas, y los dos salen de la misma
decisión: armar el vocabulario con los caracteres de un solo corpus.

El primer problema es la cobertura. Un carácter que no aparece en Tiny
Shakespeare no tiene identificador, así que `encode` tira un KeyError. La letra
`ñ`, un emoji y un carácter chino fallan igual.

El segundo problema es el largo. Un carácter da un token, así que el modelo
necesita un contexto largo para un texto corto. Y el largo se paga caro, por un
motivo que vemos con números en la sección 4.

BPE resuelve los dos. GPT-2 lo usa, y su vocabulario tiene 50257 tokens. Este
capítulo lo construye desde cero, pero antes hay que ver de qué está hecho el
piso sobre el que se apoya.

## 1. Qué es un byte y por qué el rango va de 0 a 255

Una computadora no guarda letras. Guarda números, y los guarda en grupos de
ocho interruptores, cada uno prendido o apagado. Cada interruptor de esos se
llama **bit**, y el grupo de ocho bits se llama **byte**.

Ocho interruptores de dos posiciones dan `2 * 2 * 2 * 2 * 2 * 2 * 2 * 2`
combinaciones, o sea `2^8 = 256` combinaciones distintas. Si numerás esas
combinaciones desde cero, la primera es 0 y la última es 255. De ahí sale el
rango, y no es una convención: es la cuenta.

```
00000000  ->    0
00000001  ->    1
01100001  ->   97      esto es la letra 'a'
11111111  ->  255
```

**Un byte es un número entero entre 0 y 255, sin excepciones.** No hay byte 256
ni byte negativo. Todo archivo, todo texto y todo modelo guardado en tu disco
es, mirado de cerca, una tira de esos números.

En Python los ves directo:

```python
list("a".encode("utf-8"))     # [97]
list("hola".encode("utf-8"))  # [104, 111, 108, 97]
```

## 2. Qué es UTF-8 y por qué un carácter puede ocupar más de un byte

Con 256 valores por byte alcanza para el alfabeto inglés, los dígitos y algunos
símbolos. No alcanza para el mundo: hay más de un millón de caracteres
definidos entre alfabetos, ideogramas y emojis. Un byte por carácter no entra
ni cerca.

**UTF-8 es la regla que dice cómo escribir cualquier carácter usando varios
bytes seguidos.** No es un formato de archivo ni una librería: es un acuerdo
sobre qué tiras de bytes significan qué caracteres.

### Por qué no alcanza con la solución obvia

La solución obvia sería usar siempre cuatro bytes por carácter, porque con
cuatro entran todos. El problema es el desperdicio: un texto en inglés, que es
casi todo caracteres de los primeros 128, pesaría cuatro veces más de lo
necesario, y además ningún programa viejo lo podría leer.

UTF-8 hace otra cosa. Usa **cantidad variable de bytes**: uno para los
caracteres más comunes, hasta cuatro para los más raros. Los primeros 128
valores quedan idénticos al código viejo (ASCII), así que un texto en inglés
pesa exactamente lo mismo que antes.

Mirá los cuatro casos:

```python
list("a".encode("utf-8"))    # [97]                     1 byte
list("ñ".encode("utf-8"))    # [195, 177]               2 bytes
list("日".encode("utf-8"))   # [230, 151, 165]          3 bytes
list("🤖".encode("utf-8"))   # [240, 159, 164, 150]     4 bytes
```

### Cómo sabe el lector dónde termina un carácter

Si la cantidad de bytes varía, hace falta una señal. UTF-8 la mete en los
primeros bits de cada byte. Fijate en la `ñ`, que son los bytes 195 y 177,
escritos en binario:

```
195 = 11000011    empieza con 110  ->  "arranca un carácter de 2 bytes"
177 = 10110001    empieza con 10   ->  "soy continuación del anterior"
```

El patrón sigue: `1110xxxx` anuncia tres bytes, `11110xxx` anuncia cuatro, y
`0xxxxxxx` es un carácter de un byte. Cada byte dice si es principio o
continuación, así que el que lee nunca se pierde.

Ese diseño tiene una consecuencia que vuelve en la sección 8: el byte 195 solo,
sin su compañero, es un anuncio sin cumplir. No es texto válido.

## 3. Los bytes como vocabulario base

Un **vocabulario** es la lista de todos los tokens que el modelo puede
manipular, y el identificador de un token es su posición en esa lista. El
capítulo 1 armó el vocabulario con los 65 caracteres distintos de Tiny
Shakespeare.

El **vocabulario base** es con qué arrancás antes de aprender nada. BPE no
arranca de los caracteres. Arranca de los bytes: los identificadores 0 a 255
están asignados de entrada, uno por cada valor posible de un byte, y después el
entrenamiento agrega tokens nuevos del 256 en adelante.

Lo ves en el constructor de `BPETokenizer`:

```python
self.vocab = {i: bytes([i]) for i in range(256)}
```

### Por qué 256 tokens alcanzan para cualquier idioma

Acá está el argumento completo, y es corto. Todo texto tiene una
representación en UTF-8. Toda representación en UTF-8 es una tira de bytes.
Todo byte es un número entre 0 y 255, y los 256 números ya tienen token.
Entonces no existe el texto que no se pueda escribir con esos tokens.

El japonés, el árabe, un emoji, un carácter de control y un byte nulo entran
todos, aunque el corpus de entrenamiento haya sido puro Shakespeare. Ningún
tokenizador de bytes necesita un token desconocido, porque no hay nada que le
resulte desconocido. El test `test_roundtrip_of_text_that_is_not_in_the_corpus`
prueba justo eso con `"señor"`, `"日本語"` y `"emoji: 🤖"`.

La contra la ves en la `ñ`: un carácter se convirtió en dos tokens. El texto
sobrevive, pero se alarga. Los merges están para recuperar ese costo.

## 4. Qué quiere decir comprimir acá, y por qué conviene

Comprimir, en este capítulo, quiere decir una sola cosa: **representar el mismo
texto con menos tokens**. Nada se borra y nada se aproxima. `decode(encode(t))`
tiene que devolver `t` idéntico, y hay un test que lo verifica.

El test `test_compression_ratio` lo mide con una división: caracteres del texto
dividido tokens que salieron. Si un texto de 200000 caracteres da 100000
tokens, el ratio es 2, o sea dos caracteres por token.

### Por qué menos tokens es una ventaja concreta

Acá aparece el mecanismo de attention, que ves en detalle más adelante en el
curso. Alcanza con una propiedad suya: para procesar una secuencia de largo
`T`, attention compara **cada posición con todas las demás**. Son `T * T`
comparaciones. Eso es lo que quiere decir "el costo crece con el cuadrado del
largo".

El cuadrado es traicionero porque no duplica: multiplica. Poné números a un
texto de 4000 caracteres.

```
tokens de caracteres      4000 tokens  ->  4000 * 4000  =  16.000.000 comparaciones
tokens BPE, 4 por token   1000 tokens  ->  1000 * 1000  =   1.000.000 comparaciones
```

Cuatro veces menos tokens dan dieciséis veces menos trabajo en attention. En
plata, eso son quince millones de comparaciones que no pagás, y no una sola
vez: se ahorran en cada capa del modelo y en cada paso de entrenamiento, sobre
el mismo texto y con la misma GPU. Y del lado de la inferencia, que es usar
el modelo ya entrenado para producir texto, el ahorro es todavía más directo:
un proveedor de API te factura por token, así que menos tokens es literalmente
menos factura.

Hay una segunda ventaja, gratis: con tokens más largos, un contexto de 256
posiciones abarca unos 1000 caracteres en vez de 256. El modelo ve más texto
sin pagar más.

El precio de todo esto es el vocabulario más grande. La tabla de embedding, que
es la tabla donde cada token del vocabulario guarda su vector de números, tiene
una fila por token, así que crece con el vocabulario. La capa de salida
también, porque produce un número por token posible. `vocab_size` es la perilla
que elegís vos, y elegirla es elegir dónde parar entre secuencias largas y
tablas grandes.

## 5. Qué es un algoritmo greedy

Un algoritmo **greedy** es el que en cada paso agarra lo que se ve mejor **en
ese momento**, sin evaluar si esa elección le conviene al resultado final. Toma
la decisión, no la revisa nunca, y sigue.

### Por qué eso no siempre da el mejor resultado

Mirá el caso del vuelto, con monedas de 1, 3 y 4, y un vuelto de 6:

```
greedy:  agarra la más grande que entra  ->  4, después 1, después 1  = 3 monedas
óptimo:  3 + 3                                                        = 2 monedas
```

El greedy eligió el 4 porque era el más grande disponible, y esa elección lo
dejó sin la combinación buena. Nunca vuelve atrás a revisarla.

**BPE es greedy.** En cada paso agarra el par de tokens más frecuente en ese
momento y lo fusiona, sin preguntarse si otro par le daría mejor compresión
después de veinte merges. Eso tiene dos consecuencias que vas a usar en el
código:

1. Cada merge se hace sobre el texto que dejaron los merges anteriores, así que
   los merges forman una cadena y el **orden importa**. La sección 7 vive de
   esto.
2. Cuando dos pares empatan en frecuencia hay que romper el empate con una
   regla fija, o dos corridas dan tokenizadores distintos. `max(stats,
   key=stats.get)` se queda con el primero que encontró, y `get_stats` recorre
   la lista de izquierda a derecha, así que gana el par que aparece antes en el
   texto. Es determinista, que es lo que importa.

## 6. El algoritmo de entrenamiento

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

Los once tokens se volvieron cinco. El vocabulario pasó de 4 símbolos a 7. En
el código la cuenta arranca más arriba, porque los 256 bytes están siempre: el
vocabulario va de 256 a 259, y `len(self.vocab)` termina valiendo exactamente
el `vocab_size` que pediste. Ese es el trade-off de BPE, y el argumento
`vocab_size` lo controla.

En el código no hay letras: `Z` es el identificador 256, `Y` es el 257 y `X` es
el 258, porque los primeros 256 ya están tomados por los bytes. El test
`test_encode_applies_the_merges_in_order` entrena con este mismo texto y espera
exactamente `{(a, a): 256, (256, a): 257, (257, b): 258}`.

En el segundo paso hubo un empate que el ejemplo pasa por arriba: `(Z, a)` y
`(a, b)` aparecen dos veces cada uno. Ganó `(Z, a)` porque aparece antes en la
lista, que es la regla de la sección 5.

Visto de arriba, el capítulo entero es esta cañería:

```
  texto  --encode utf-8-->  bytes  --merges en orden-->  tokens
                                                            |
  texto  <--decode utf-8--  bytes  <--vocab[id] por id------+
```

**Ojo con el solapamiento.** En `[1, 1, 1]` el par `(1, 1)` aparece dos veces,
pero un merge puede reemplazar solo uno. El resultado es `[Z, 1]`, no `[Z, Z]`
ni `[Z]`. Un loop con un índice explícito lo maneja bien, porque después de una
coincidencia avanza dos posiciones en vez de una. Un loop que avanza siempre de
a una posición se come el token del medio dos veces, y ahí sale `[Z, Z]`. El
test `test_merge_handles_the_overlap` mira justo eso.

`train` hace `vocab_size - 256` merges, así que un `vocab_size` menor que 256
no tiene sentido: los bytes solos ya ocupan 256 lugares y no se pueden sacar.
Ese caso se rechaza con un `ValueError` antes de tocar el texto, y
`test_train_rejects_a_vocabulary_under_256` lo pide.

Y `train` tiene que aguantar el texto sin pares. Si le pasás `"a"`, el primer
`get_stats` devuelve un dict vacío y no hay nada que fusionar: el loop corta ahí
en vez de romper, y el tokenizador queda con cero merges. Es lo que verifica
`test_train_stops_early_when_no_pair_is_left`.

## 7. encode necesita el orden de los merges

Los merges no son un conjunto. Son una lista ordenada, y `encode` tiene que
aplicarlos en el mismo orden que `train`.

La razón es la naturaleza greedy del entrenamiento: los merges de después se
construyen sobre los de antes. El token `X` del ejemplo de arriba significa
`aaab`, pero solo porque `Y` y `Z` ya existen. Un `encode` que aplique `X`
primero da otro resultado, y el modelo nunca vio esos identificadores durante
el entrenamiento.

Entonces `encode` repite un solo paso: de todos los pares que aparecen en la
lista actual, buscá el de menor índice de merge y aplicalo. Frená cuando ningún
par de la lista esté en la tabla de merges.

Un detalle de implementación que ahorra un `if` adentro del loop: si a los
pares que no están en la tabla les asignás `float("inf")` como índice, `min()`
nunca los elige, y te queda un solo chequeo al final para cortar.

## 8. Lo que decode tiene que aguantar

`decode` concatena los bytes de cada token y después decodifica UTF-8. Acá
vuelve lo de la sección 2: una lista cualquiera de identificadores puede dar
una secuencia de bytes que no sea UTF-8 válido, porque un carácter de varios
bytes puede quedar cortado por la mitad. El byte 195 solo anuncia un carácter
de dos bytes que nunca llega. Durante la generación, el modelo puede producir
justo eso, y lo hace seguido cuando todavía está poco entrenado.

La respuesta estándar es `errors="replace"`, que pone el carácter `U+FFFD` (se
ve como `�`) donde estaban los bytes inválidos. La alternativa, una excepción,
corta la generación del modelo por un motivo que no tiene ningún interés.

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
