# Capítulo 1: el pipeline de datos

## 1. Qué es un modelo de lenguaje

Todo el curso construye una sola máquina, y conviene saber desde el principio
qué hace esa máquina. No entiende, no razona, no busca. Hace una única cosa.

**Un modelo de lenguaje recibe un pedazo de texto y devuelve, para cada
símbolo posible que podría venir después, un número que dice qué tan probable
es ese símbolo.**

Poné que el texto hasta acá es `El gato subió al`. Si el vocabulario tuviera
solo cuatro palabras, la salida del modelo sería algo así:

```
'techo'  0.62
'auto'   0.21
'gato'   0.14
'.'      0.03
```

Esos cuatro números son probabilidades: van de 0 a 1 y suman 1. El modelo no
elige una. Reparte la creencia entre todas las opciones, y quien lo usa decide
después qué hacer con ese reparto: quedarse con la más alta, o sortear una al
azar respetando esas proporciones (eso último es el muestreo, y lo vas a
escribir en el capítulo 8).

Ojo con un detalle que aparece en todos los capítulos siguientes: la última
capa del modelo no produce probabilidades. Produce un puntaje crudo por cada
símbolo, que puede ser negativo o gigante y que no suma nada en particular.
Esos puntajes se llaman **logits**. La función que los convierte en
probabilidades es el **softmax**, que escribiste en el capítulo 0. Cuando en
este capítulo digo "el modelo predice el próximo token", quiero decir esa
cadena completa: logits, softmax, probabilidades.

Y "predecir el próximo token", entonces, no quiere decir adivinar la respuesta
correcta. Quiere decir repartir bien. Un modelo bueno le da probabilidad alta
a lo que efectivamente vino después, en el texto real, una y otra vez, en
millones de posiciones. Entrenar es empujar esos números en esa dirección.
Nada más que eso.

## 2. Qué es un token y qué es un vocabulario

El texto es una tira de caracteres. La red neuronal, adentro, hace
multiplicaciones y sumas. No podés multiplicar la letra `a` por 0.37. Así que
antes de tocar el modelo hay que convertir el texto en números.

**Un token es la unidad mínima de texto que el modelo ve, y el vocabulario es
la lista completa de tokens distintos que existen para ese modelo.**

En este capítulo un token es un carácter. `F` es un token, `i` es un token, el
espacio es un token, el salto de línea también.

### Por qué no alcanza con usar el código ASCII

La idea obvia es no inventar nada y usar el número que cada carácter ya tiene:
`ord('a')` da `97`, `ord('F')` da `70`. Listo, texto convertido en números.

No sirve, por dos motivos.

El primero es el tamaño. El modelo tiene que producir un logit por cada token
posible, así que la capa final tiene tantas salidas como tamaño tenga el
vocabulario. Si usás Unicode entero, son más de un millón de salidas para un
texto que en realidad usa 65 símbolos. Pagarías una matriz enorme llena de
posiciones que nunca se usan.

El segundo es que los identificadores tienen que ser **densos y arrancar en
cero**, porque se usan como índice. En el capítulo 3 vas a ver que el modelo
guarda una tabla con una fila por token, y busca la fila número `i`. Si tus
identificadores son `10`, `70` y `97`, necesitás una tabla de 98 filas para
guardar 3 cosas.

Entonces el tokenizador arma su propio mapa, chico y compacto: agarra los
caracteres que de verdad aparecen en el corpus, los ordena, y usa la posición
en esa lista ordenada como identificador. El **corpus** es el texto con el que
entrenás, todo junto y como una sola tira; acá es Tiny Shakespeare, un archivo
de 1115394 caracteres con las obras de Shakespeare una atrás de la otra.

Con el texto `First`:

```
caracteres distintos, ordenados:   ['F', 'i', 'r', 's', 't']
posición en la lista:                0    1    2    3    4

encode("First")  ->  [0, 1, 2, 3, 4]
decode([0, 1, 2, 3, 4])  ->  "First"
```

Ese `sorted(set(text))` produce `chars`, y de ahí salen los dos diccionarios:
`stoi` lleva de carácter a entero, `itos` de entero a carácter. Uno es el
inverso del otro, y el test lo verifica: si codificás y decodificás, tenés que
recuperar el texto idéntico.

Tiny Shakespeare tiene 65 caracteres distintos. Así que el vocabulario es de
65 tokens.

### Por qué el orden importa

**`sorted(set(text))` da el mismo resultado en todas las corridas.
`set(text)` solo, no.** El orden de iteración de un `set` de Python no es algo
con lo que puedas contar entre procesos.

Y esto no es prolijidad. Un modelo entrenado y guardado en disco tiene adentro
identificadores, no caracteres. Aprendió que el token `41` va seguido del
token `12`. Si mañana levantás el modelo con un tokenizador que armó el mapa
en otro orden, el `41` ahora es otra letra, y el modelo escupe basura sin dar
ningún error. Fallar así es peor que fallar fuerte.

### Caracteres contra BPE

Un carácter por token es la opción más simple, no la mejor. GPT-2 usa un
vocabulario de 50257 tokens, donde cada token es un pedazo de palabra, armado
con un algoritmo que se llama BPE. Eso lo construís en el capítulo 2. Los dos
tamaños se compensan entre sí:

| | Tokens de caracteres | Tokens BPE |
|---|---|---|
| Vocabulario | 65 | 50257 |
| Tokens para el mismo texto | muchos | unas 4 veces menos |
| Tabla de embedding | chica | grande |
| Un contexto de 256 tokens abarca | unos 256 caracteres | unos 1000 caracteres |

Dos nombres de esa tabla, para que no te agarren de sorpresa. La **tabla de
embedding** es la tabla con una fila por token que mencioné arriba, la que se
indexa con el identificador; la escribís en el capítulo 3. Y **attention** es
el mecanismo que deja que cada posición mire a las anteriores, que es el
capítulo 4.

Un vocabulario chico te da secuencias largas, y el costo de attention crece
con el cuadrado del largo de la secuencia: el doble de tokens cuesta cuatro
veces más. Un vocabulario grande te da secuencias cortas, pero también una
tabla de embedding grande y una capa de salida grande. Es una decisión de
ingeniería, con costo de los dos lados.

## 3. Self-supervised: por qué acá no hay etiquetas

En el aprendizaje supervisado clásico necesitás dos columnas. Una foto y la
palabra "gato". Un mail y la marca "spam". Esa segunda columna es la
**etiqueta**, y sale de gente que se sentó a escribirla. Esa gente es cara y
es lenta, y por eso los datasets etiquetados son chicos.

**Self-supervised quiere decir que la etiqueta ya está adentro del dato, así
que no hay que producirla aparte.**

En un modelo de lenguaje la etiqueta de cada posición es, simplemente, el
token que viene después. El texto es su propia respuesta correcta:

```
texto:      F  i  r  s  t
entrada     F           ->  etiqueta: i
entrada     F  i        ->  etiqueta: r
entrada     F  i  r     ->  etiqueta: s
entrada     F  i  r  s  ->  etiqueta: t
```

Nadie etiquetó nada. El texto ya estaba escrito. Por eso los datos de
entrenamiento de este curso no son una tabla de entradas y etiquetas: son una
sola secuencia larga de tokens, y las etiquetas se leen corriendo esa misma
secuencia una posición.

Esto es lo que hace posible entrenar con internet entero. Todo texto que
exista es dataset, sin que nadie lo anote.

## 4. Tensores, shapes, y qué quiere decir (B, T)

**Un tensor es un arreglo de números, todos del mismo tipo, con una forma
conocida.** La palabra suena a matemática pesada, pero para lo que necesitás
acá alcanza con esto: es una lista de números, o una lista de listas, o una
lista de listas de listas.

La cantidad de niveles se llama **dimensiones**, y la lista de tamaños de cada
nivel se llama **shape**.

Un tensor 1-D es una tira. Su shape es una tupla de un solo número:

```python
data = torch.tensor([0, 1, 2, 3, 4])
data.shape        # torch.Size([5])  cinco elementos, un solo nivel
data[2]           # 2                un índice alcanza para llegar a un número
```

`torch.Size` es una tupla común, así que `data.shape` se lee y se compara como
`(5,)`.

Un tensor 2-D es una tabla, con filas y columnas. Su shape tiene dos números:

```python
x = torch.tensor([[10, 11, 12],
                  [20, 21, 22]])
x.shape           # torch.Size([2, 3])  dos filas, tres columnas
x[1, 0]           # 20                  hacen falta dos índices
```

El primer número del shape siempre es el nivel de más afuera. `(2, 3)` es
"dos cosas, y cada una tiene tres números adentro".

En este capítulo aparecen los dos. El corpus entero es un tensor **1-D**: una
sola tira de identificadores, del primer carácter del archivo al último, con
shape `(1115394,)` para Tiny Shakespeare. Y lo que le pasás al modelo en cada
paso es un tensor **2-D** con shape `(B, T)`:

- `B` es `batch_size`, la cantidad de bloques que van juntos.
- `T` es `block_size`, la cantidad de tokens de cada bloque.

Con `B = 4` y `T = 8`, `x` es una tabla de 4 filas por 8 columnas: cuatro
pedazos distintos del texto, sacados de cuatro lugares al azar, apilados uno
arriba del otro. `x[2, 5]` es el sexto token del tercer bloque.

En el resto del curso vas a ver shapes de tres y cuatro números, como
`(B, T, C)` cuando cada token se convierte en un vector de `C` números. La
lectura es siempre la misma: de afuera hacia adentro.

## 5. Qué es un batch y por qué no se entrena de a un ejemplo

Entrenar es esto: le mostrás un ejemplo al modelo, medís cuánto se equivocó, y
movés cada parámetro un poquito en la dirección que hace bajar ese error. Los
parámetros son los números que el modelo ajusta mientras aprende, y son
millones. La medida de cuánto se equivocó es el **loss**, un solo número:
más alto es peor. Y esa dirección para mover cada parámetro es el
**gradiente**, que calculaste a mano en el capítulo 0. Pensalo como la
pendiente del terreno abajo de tus pies: te dice para dónde se baja, ahí donde
estás parado.

La solución obvia sería procesar un ejemplo, dar un paso, procesar el
siguiente, dar otro paso. No se hace así, por dos motivos.

**El motivo estadístico.** El gradiente de un solo ejemplo es la pendiente del
error *de ese ejemplo*, no del texto en general. Un ejemplo dice "subí el peso
de la `u` después de la `q`". El siguiente, sacado de otra parte del corpus,
puede decir lo contrario con la misma fuerza. Si vas ejemplo por ejemplo, el
modelo zigzaguea siguiendo el ruido. Con cinco ejemplos que proponen
`+0.9, -0.7, +1.1, -0.5, +0.8`, el promedio es `+0.32`: mucho más chico que
cualquiera de ellos, y mucho más confiable, porque las contradicciones se
cancelan y queda lo que los cinco tienen en común.

**El motivo de hardware.** Una GPU tiene miles de unidades que hacen la misma
cuenta al mismo tiempo. Pasarle una fila la deja casi entera sin trabajo.
Pasarle 32 filas tarda casi lo mismo que pasarle una. Procesar de a uno no es
más barato, es la misma plata tirada 32 veces.

**Un batch es un grupo de ejemplos que se procesan juntos en un solo paso, y
cuyos gradientes se promedian antes de mover un solo parámetro.**

Por eso `get_batch` devuelve `B` bloques y no uno. Y por eso los tensores del
curso tienen esa primera dimensión `B` adelante de todo: es la dimensión
"cuántos ejemplos van juntos", y casi ninguna operación del modelo la mira,
solo la arrastra.

## 6. Overfitting, y para qué está el conjunto de validación

Un modelo con muchos parámetros tiene dos maneras de bajar el error sobre el
texto que le mostrás. Puede aprender cómo funciona el idioma, o puede
memorizar el texto tal cual. Las dos bajan el número, y desde adentro se ven
igual.

**Sobreajustar (overfitting) es cuando el modelo mejora en el texto que ya
vio, y al mismo tiempo empeora en texto nuevo.** Es un alumno que se aprende
de memoria las respuestas del examen del año pasado.

Como el loss de entrenamiento no distingue esos dos casos, hace falta una
segunda medición. Apartás un pedazo del texto antes de empezar, el modelo
nunca entrena con ese pedazo, y cada tanto medís el error ahí. Ese pedazo es
el **conjunto de validación**, y es tu único instrumento honesto:

```
loss de entrenamiento baja, loss de validación baja   ->  está aprendiendo
loss de entrenamiento baja, loss de validación sube   ->  está memorizando
```

Sin ese segundo número no tenés forma de saber en cuál de los dos casos estás.

### Por qué el split no puede ser al azar

`train_val_split` corta la secuencia en dos partes contiguas y manda el último
10 por ciento a validación. Nada de mezclar. Y acá la razón es propia de los
datos secuenciales, no una manía.

Los ejemplos de entrenamiento son ventanas sobre el texto, y las ventanas se
superponen. Un split que reparte posiciones al azar manda la ventana que
arranca en la posición 100 a entrenamiento, y la que arranca en la 101 a
validación. Con `T = 256`, esas dos ventanas comparten 255 de sus 256
caracteres.

O sea que el modelo entrenó con casi exactamente el texto que después le vas a
tomar. El loss de validación mide memoria en vez de generalización, te da un
número hermoso, y te deja sin instrumento justo cuando más lo necesitás. Un
split contiguo garantiza que el texto de validación nunca pasó por el
entrenamiento.

## 7. La idea central: un bloque de T tokens son T ejemplos

Esta es la parte que sorprende, y la que hace que este capítulo exista.

**Un bloque de 256 tokens no es un ejemplo de entrenamiento. Son 256 ejemplos
de entrenamiento adentro de un solo tensor.**

La razón es la que viste en la sección 3: cada posición del bloque es un
prefijo con su etiqueta al lado. Tomá el bloque `[F, i, r, s, t]`. El modelo
ve esto:

```
input                 target
[F]                -> i
[F, i]             -> r
[F, i, r]          -> s
[F, i, r, s]       -> t
```

Cuatro predicciones distintas, no una. Y no cuestan cuatro pasadas: la
**máscara causal** del capítulo 4 hace que la posición `t` pueda leer las
posiciones de `0` hasta `t` y nada de lo que viene después, así que un solo
forward (una pasada del dato por el modelo, de la entrada a los logits)
produce las cuatro salidas a la vez, cada una mirando solo su propio prefijo.

### El ejemplo numérico completo

Por eso el código no construye esas cuatro filas. Construye dos tensores:

```
x = data[i     : i + T]        el bloque
y = data[i + 1 : i + T + 1]    el mismo bloque, corrido una posición
```

Seguilo con números. Con el texto `First`, el tokenizador de la sección 2 y
`T = 4`:

```
data      = [0, 1, 2, 3, 4]        que es  F  i  r  s  t
offset i  = 0

x = data[0:4] = [0, 1, 2, 3]       F  i  r  s
y = data[1:5] = [1, 2, 3, 4]       i  r  s  t
```

Ahora poné los dos, uno arriba del otro, y leelos por columna:

```
posición t:      0     1     2     3
x[t]:            0     1     2     3        F     i     r     s
y[t]:            1     2     3     4        i     r     s     t
```

La columna `t = 0` dice: habiendo visto `F`, lo que sigue es `i`. La columna
`t = 2` dice: habiendo visto `F i r` (porque el modelo, en la posición 2, ve
todo lo anterior), lo que sigue es `s`. Cuatro columnas, cuatro ejemplos, dos
tensores chiquitos.

Fijate el conteo, que es la clave de la sección: `x` tiene `T = 4` tokens y da
4 predicciones. El quinto token del texto, la `t`, no está en `x`: es la
etiqueta de la última posición, y vive en `y`. Por eso un bloque de `T` tokens
da exactamente `T` ejemplos, y por eso `y` necesita leer una posición más allá
que `x`.

Escrito como regla: **`y[t]` es la respuesta correcta para el prefijo que
termina en `x[t]`**. Los tests del capítulo verifican exactamente eso con
`assert_close(y[:, :-1], x[:, 1:])`, o sea que `y` sin su última columna es
`x` sin su primera.

Y esto multiplica por `T` todo lo que aprende cada paso. Un tensor de shape
`(B, T)` contiene `B * T` predicciones:

```
B = 32 bloques
T = 256 tokens por bloque
32 * 256 = 8192 predicciones en un solo paso de entrenamiento
```

Más de ocho mil ejemplos por el precio de mover 32 ventanas. Un modelo de lenguaje
saca muchísima más señal por token de la que parece a primera vista.

### El offset válido más grande

Como `y` lee una posición más allá que `x`, el offset `i` tiene que cumplir:

```
i + T + 1 <= len(data)        o sea        i <= len(data) - T - 1
```

Acá hay una trampa de índices que vale la pena mirar despacio, porque
`torch.randint(high, ...)` **no incluye** `high`: te devuelve valores de `0` a
`high - 1`. Entonces el argumento correcto es `len(data) - T`, que produce
como máximo el offset `len(data) - T - 1`, que es justo el que querías.

Verificalo con el caso más ajustado que hay, que es un test del capítulo:

```
len(data) = 20        T = 19
high = 20 - 19 = 1    ->  randint devuelve siempre 0
x = data[0:19]        ->  llega hasta el token 18
y = data[1:20]        ->  llega hasta el token 19, el último de todos
```

Justo, sin sobrar ni faltar. Si pasás `len(data)` te vas de la tira: el
offset más grande queda `T` posiciones afuera de los datos. Si pasás
`len(data) - T - 1` te quedás corto y el último bloque del corpus no se
entrena nunca; en este caso extremo `high` da `0` y `randint` directamente
falla.

### Y por qué el generator

`get_batch` recibe un `torch.Generator` y se lo pasa a `torch.randint`. Un
generator es la fuente de números al azar: con la misma semilla produce
siempre la misma secuencia. Pasarlo hace que la misma seed te dé el mismo
batch, y eso es lo que te permite comparar dos corridas y saber que la
diferencia la causó tu cambio y no la suerte del sorteo.

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
