# Glosario

Cada término que usa el curso está definido acá, con la explicación completa y
no con una referencia a otro capítulo. Si llegaste desde el capítulo 9 y
necesitás saber qué es un logit, la entrada te alcanza sin volver al 0. Las
entradas se citan entre sí, así que podés tirar del hilo hasta donde te sirva.

## ablación

Un experimento que le saca una pieza al modelo, lo vuelve a entrenar, y mide
cuánto se pierde. Es la única forma honesta de saber si esa pieza hacía algo o
estaba de adorno. El capítulo 13 corre tres: top-1 contra top-2, cambiar la
cantidad de expertos, y entrenar sin el loss auxiliar. La última es la más
interesante, porque muestra el colapso del router en números.

## AdamW

El optimizador que usa el curso para decidir cuánto mover cada parámetro. En
vez de aplicarle el mismo learning rate a todos, guarda dos promedios móviles
por parámetro: uno del gradiente y otro del gradiente al cuadrado. Con esos dos
números le da un paso más grande a los parámetros cuyo gradiente es chico y
constante, y un paso más chico a los que oscilan. La `W` es por *decoupled
weight decay*: el término que empuja los pesos hacia cero se aplica aparte del
gradiente, y no adentro de él.

## ASCII

La tabla vieja que le da un número del 0 al 127 a las letras del inglés, a los
dígitos y a unos pocos símbolos y caracteres de control. `ord('a')` da `97`
porque así lo fijó ASCII. No le alcanza ni para la `ñ`, y por eso existe UTF-8,
que la contiene: los primeros 128 valores de UTF-8 son exactamente los de
ASCII, así que un texto en inglés pesa lo mismo en los dos.

## attention

La operación que le permite a cada posición de la secuencia mirar a las otras y
armar un promedio ponderado de lo que encuentra. Cada posición emite una
consulta, cada posición ofrece una clave, y el producto entre consulta y clave
da un score: cuánto le interesa a una lo que tiene la otra. Esos scores pasan
por un softmax, y los pesos que salen se usan para promediar los valores. Es
todo el mecanismo: comparar, normalizar, promediar.

## backward

El recorrido que va del loss hacia atrás, hasta cada parámetro, calculando el
gradiente de cada uno con la regla de la cadena. Va en el sentido contrario al
forward y reusa los valores intermedios que el forward guardó. En PyTorch lo
disparás con `loss.backward()`, y el capítulo 0 te hace escribir a mano el de
una capa lineal para que veas qué hace por dentro. Después del backward, cada
parámetro tiene guardado su `.grad` y el optimizador puede dar el paso.

## balanceo de carga

El problema de que los expertos de una capa MoE reciban una cantidad parecida
de tokens. Sin nada que lo impida, el router entra en un círculo vicioso: el
experto que arranca un poco mejor recibe más tokens, entrena más, mejora más, y
termina llevándose todo. Eso se llama colapso, y deja al modelo con ocho
expertos de los cuales uno solo aprendió algo. El capítulo 10 lo mide y lo
arregla con el loss auxiliar.

## baseline

Un modelo simple que corrés primero para tener contra qué comparar. En este
curso el baseline es el modelo de bigramas del capítulo 3, que predice el
próximo token mirando solo el anterior. Sirve para dos cosas: te da un número de
loss que cualquier cosa mejor tiene que superar, y te obliga a tener el pipeline
de datos y de entrenamiento funcionando antes de agregar attention.

## batch

El grupo de ejemplos que el modelo procesa junto, en un solo forward y un solo
backward. Se agrupan por dos motivos: la GPU hace 32 multiplicaciones de
matrices casi al mismo costo que una, y el gradiente promediado sobre muchos
ejemplos tiene menos ruido que el de uno solo. Acá un batch es un tensor de
shape `(B, T)`, con `B` bloques de `T` tokens cada uno. Con `B = 32` y
`T = 256` son 8192 predicciones en un solo paso.

## bigrama

Un modelo que predice el próximo token mirando únicamente el token anterior, sin
nada de contexto más atrás. Se implementa con una tabla de `V` filas por `V`
columnas, donde la fila `i` guarda los logits del próximo token cuando el actual
es `i`. No hay attention ni capas ocultas: la tabla *es* el modelo. Es el punto
de partida del capítulo 3, y es la cota inferior contra la que se mide todo lo
que viene.

## bloque

La unidad que se repite para armar el transformer. Cada uno tiene dos partes:
un multi-head attention y una capa feed-forward, y cada parte va envuelta en un
LayerNorm adelante y una conexión residual alrededor. El modelo del curso apila
6 bloques idénticos en estructura, con parámetros propios cada uno. La versión
MoE cambia una sola cosa adentro del bloque: la capa feed-forward se reemplaza
por una capa Mixture of Experts.

## BPE

Sigla de *Byte Pair Encoding*, el algoritmo con el que se arma el vocabulario de
GPT-2. Arranca con los 256 bytes posibles como vocabulario, cuenta qué par de
tokens adyacentes aparece más veces en el corpus, le da un identificador nuevo a
ese par, y reemplaza todas sus apariciones. Repite hasta llegar al tamaño de
vocabulario que pediste. El resultado es una lista ordenada de merges, y ese
orden es parte del tokenizador: `encode` tiene que aplicarlos en la misma
secuencia en que `train` los creó.

## broadcast

La regla de PyTorch que estira automáticamente una dimensión de tamaño 1 para
que dos tensores de shapes distintos se puedan operar. Un tensor `(32, 65)`
menos uno `(32, 1)` funciona: la columna única se repite 65 veces. Un tensor
`(32, 65)` menos uno `(32,)` falla o, peor, hace la resta sobre la dimensión
equivocada. Por eso el `keepdim=True` de `max` y de `sum` aparece tanto en el
capítulo 0.

## byte

Un número entero entre 0 y 255, y la unidad con la que la computadora guarda
todo. Sale de agrupar ocho bits, o sea ocho interruptores de dos posiciones:
`2^8` da 256 combinaciones, numeradas de 0 a 255. No existe el byte 256 ni el
byte negativo. BPE lo usa como piso: su vocabulario base son los 256 bytes
posibles, así que ningún texto le queda afuera.

## cabeza

Una unidad de attention independiente, con sus propias matrices de consulta,
clave y valor. Cada una aprende a mirar una relación distinta: una puede seguir
concordancias de género, otra el paréntesis que quedó abierto. El modelo del
curso usa 6 cabezas de 64 dimensiones cada una, y 6 por 64 da las 384
dimensiones del modelo. La suma de las cabezas cuesta lo mismo que una cabeza
grande, y aprende más.

## capa lineal

La operación `y = x @ W + b`, o sea multiplicar por una matriz de pesos y sumar
un vector de sesgo. Es la pieza con parámetros más común de todo el modelo:
está adentro del attention, adentro del feed-forward, y en la salida. Con `x` de
shape `(N, in)` y `W` de shape `(in, out)`, la salida queda `(N, out)`. El
capítulo 0 te hace derivar sus tres gradientes a mano.

## conjunto de validación

La parte del texto que apartás y que el modelo nunca ve durante el
entrenamiento. Sirve para responder la única pregunta que importa: si el modelo
aprendió algo general o se memorizó el corpus. El curso reserva el último 10 por
ciento del texto, y el corte es contiguo y no aleatorio. Un corte aleatorio
pondría ventanas casi idénticas de los dos lados, y el número de validación
saldría mucho mejor de lo que la realidad justifica.

## contexto / block_size

La cantidad máxima de tokens que el modelo puede mirar a la vez para predecir el
siguiente. Acá vale 256, así que la posición 300 no puede ver la posición 10:
esa información quedó afuera de la ventana. El costo del attention crece con el
cuadrado de este número, porque cada posición compara contra todas las
anteriores. Duplicar el contexto cuadruplica el trabajo del attention, y por eso
no se agranda gratis.

## corpus

El texto con el que entrenás, todo junto y tratado como una sola tira. Acá es
Tiny Shakespeare, y el curso lo convierte en un tensor 1-D de tokens del que
`get_batch` saca ventanas al azar. El corpus decide dos cosas que después no se
pueden cambiar sin volver a entrenar: el vocabulario del capítulo 1 y los
merges del capítulo 2.

## cosine schedule

La curva por la que baja el learning rate a lo largo del entrenamiento, con la
forma de la primera mitad de un coseno. Arranca en el valor máximo, baja despacio
al principio, rápido en el medio, y despacio otra vez cerca del final, hasta
terminar en un valor mínimo cercano a cero. La idea es dar pasos grandes
mientras el modelo está lejos, y pasos finos cuando ya está cerca del fondo del
valle. Va después del warmup: primero se sube, después se baja siguiendo esta
curva.

## cross entropy

El loss que se usa para entrenar modelos de lenguaje. Se calcula así: mirás qué
probabilidad le dio el modelo al token que de verdad venía, y tomás el logaritmo
negativo de ese número. Si le dio probabilidad 1, el loss es `-log(1) = 0`,
perfecto. Si le dio 0.001, el loss es `6.9`, pésimo. El castigo crece muy rápido
cuando la probabilidad se acerca a cero, y esa es la propiedad que hace que el
modelo aprenda a no descartar del todo lo que después ocurre.

## denso

Un modelo donde todos los parámetros participan en el cálculo de todos los
tokens. GPT-2 es denso: cada token que entra pasa por las mismas matrices que
cualquier otro. Es lo opuesto a sparse, y es la arquitectura de la parte 2 del
curso. La comparación del capítulo 12 pone un modelo denso contra uno MoE con la
misma cantidad de parámetros activos.

## derivada

Cuánto se mueve la salida de una función cuando movés su entrada un poquito. Si
`f(w) = w * w` y estás parado en `w = 3`, mover `w` a `3.01` sube la salida de
`9` a `9.0601`: son `0.0601` de salida por `0.01` de entrada, así que la
derivada vale `6`. El signo te dice para qué lado se mueve la salida y el
tamaño te dice cuánto le importa esa entrada. Cuando hay más de una variable,
la versión que se usa es la derivada parcial.

## derivada parcial

Cuánto cambia el resultado de una función cuando movés una sola de sus
variables un poquito y dejás todas las demás quietas. Si `f(x, y) = 3x + y²`,
al mover `x` el resultado cambia 3 por unidad, así que la derivada parcial
respecto de `x` es 3. Al mover `y` cambia `2y`, o sea que depende de dónde
estés parado. Un modelo tiene 10 millones de variables, y el entrenamiento
necesita justo eso: cuánto empeora o mejora el loss al mover cada una.

## descenso por gradiente

El método completo de entrenamiento, en tres pasos que se repiten: calculás el
loss, calculás el gradiente, y movés cada parámetro un poquito en la dirección
contraria a su gradiente. La dirección contraria porque el gradiente apunta a
donde el loss sube, y vos querés que baje. El "poquito" es el learning rate, y
elegirlo mal arruina el entrenamiento en cualquiera de las dos direcciones.
AdamW es una versión afinada de esta misma idea.

## dispatch

El paso de una capa MoE que agarra cada token y lo manda físicamente al experto
que le tocó. La versión naive es un loop en Python: para cada experto, junto los
tokens que le corresponden y los proceso. La versión vectorizada hace lo mismo
con operaciones de índices sobre tensores, sin loop, y es la que corre rápido en
la GPU. El test del capítulo 11 exige que las dos den exactamente el mismo
resultado.

## embedding

La fila de números que representa a un token adentro del modelo. La tabla de
embeddings es una matriz de `V` filas por `d` columnas, donde `V` es el tamaño
del vocabulario y `d` la dimensión del modelo, acá 384. Convertir un token en su
vector es buscar la fila: el token 17 usa la fila 17. Esos números arrancan al
azar y los aprende el entrenamiento, así que dos tokens que aparecen en
contextos parecidos terminan con filas parecidas.

## epoch

Una pasada completa por todos los datos de entrenamiento. Es una unidad común en
otros problemas, pero este curso no la usa: `get_batch` toma ventanas al azar
del corpus, así que no hay un momento claro en el que se hayan visto todos los
datos exactamente una vez. Acá se cuenta en pasos de entrenamiento, o sea
cuántas veces se actualizaron los parámetros. Es la unidad honesta cuando los
ejemplos se muestrean al azar y se superponen.

## escalar

Un tensor de cero dimensiones, o sea un número suelto, con shape `()`. El loss
es el caso típico: por más grande que sea el batch, el resultado de
`cross_entropy` es un escalar, porque el entrenamiento necesita un solo número
del que salga el gradiente. No es lo mismo que un tensor de shape `(1,)`, que
tiene una dimensión de tamaño 1. `.item()` te lo pasa a un float de Python.

## experto

Cada una de las redes feed-forward que conviven adentro de una capa Mixture of
Experts. Todas tienen la misma estructura y parámetros distintos, y cada token
pasa solo por algunos. El modelo MoE del curso usa 8 expertos con top-2, así que
cada token usa 2 de los 8. La palabra sugiere que cada uno se especializa en un
tema, y en la práctica la especialización que aparece es más rara y menos
interpretable que eso.

## factor de capacidad

El multiplicador que fija cuántos tokens puede aceptar cada experto en un batch.
La cuenta es `capacidad = factor * tokens_del_batch * k / cantidad_de_expertos`,
o sea el reparto perfectamente parejo por el factor. Con factor 1.0 no hay
margen: en cuanto un experto recibe uno más que su parte exacta, los que sobran
se caen. Con 1.25 hay un 25 por ciento de colchón, que es lo que se usa en la
práctica porque el reparto real nunca es perfecto.

## feed-forward

La red de dos capas lineales que va en la segunda mitad de cada bloque. La
primera capa expande de 384 a 1536 dimensiones, pasa por GELU, y la segunda
vuelve a 384. Es donde vive la mayor parte de los parámetros del modelo, y donde
el modelo hace su procesamiento por posición: cada token pasa por acá sin mirar
a los demás. La capa Mixture of Experts reemplaza exactamente esta pieza.

## forward

El recorrido que va de los tokens de entrada hasta los logits de salida y el
loss, pasando por todas las capas en orden. Es el modelo haciendo su trabajo. En
PyTorch lo escribís vos en el método `forward` de cada módulo, y llamar al
módulo lo ejecuta. Durante el entrenamiento, el forward además guarda los
valores intermedios que el backward va a necesitar.

## GELU

La función de activación que usa GPT-2 adentro de la capa feed-forward, y la que
le da al modelo la capacidad de aprender cosas que no sean una línea recta. Sin
una activación, dos capas lineales encadenadas colapsan en una sola capa
lineal, y todo el modelo se vuelve una multiplicación de matrices. GELU se
parece a ReLU pero es suave: en vez de cortar seco en cero, la transición es
gradual, y los valores levemente negativos pasan atenuados en lugar de morir.
Esa suavidad le da al gradiente algo con qué trabajar en la zona del cero.

## GPT-2

El modelo de lenguaje que publicó OpenAI en 2019, y el objetivo de la parte 2
del curso. Es un transformer decoder-only: bloques apilados, attention con
máscara causal, y predicción del próximo token. Su vocabulario tiene 50257
tokens, armado con BPE. El test más fuerte del curso está en el capítulo 7:
cargás los pesos reales de `gpt2` en tu propio modelo y tienen que salir los
mismos logits que la referencia, con `atol=1e-4`.

## grad clipping

Un tope al tamaño total del gradiente antes de que el optimizador dé el paso. El
tamaño se mide con la norma: elevás cada gradiente al cuadrado, los sumás todos
y sacás la raíz, que es el largo del vector. Si ese largo pasa el umbral,
típicamente `1.0`, se escalan todos los gradientes por el mismo factor para que
quede justo en el tope. Sirve para
un problema concreto: cada tanto aparece un batch raro que produce un gradiente
gigante, el modelo pega un salto enorme, y el loss se dispara o se vuelve `nan`.
La dirección del paso no cambia, solo su largo.

## gradiente

El vector con todas las derivadas parciales de una función, una por cada
variable. Si el loss depende de 10 millones de parámetros, el gradiente es una
lista de 10 millones de números, y cada uno te dice cuánto empeora el loss al
subir ese parámetro. Apunta en la dirección donde la función sube más rápido,
así que para bajar el loss te movés en la dirección contraria. Ese es todo el
entrenamiento.

## greedy

Un algoritmo que en cada paso agarra lo que se ve mejor en ese momento y nunca
revisa esa decisión. Es rápido y simple, y no siempre da el mejor resultado
final: con monedas de 1, 3 y 4, un vuelto de 6 le sale `4 + 1 + 1`, cuando
`3 + 3` era mejor. BPE es greedy, porque fusiona el par más frecuente de la
vuelta actual sin preguntarse qué le conviene veinte merges después. De ahí
sale que los merges formen una cadena en la que el orden importa.

## hiperparámetro

Un número que elegís vos antes de entrenar, y que el entrenamiento no ajusta. El
learning rate, la cantidad de capas, el tamaño del batch, la cantidad de
expertos y el factor de capacidad son hiperparámetros. La diferencia con un
parámetro es quién lo mueve: al parámetro lo mueve el gradiente, al
hiperparámetro lo movés vos entre una corrida y la siguiente.

## inferencia

Usar un modelo ya entrenado para producir texto, sin tocar los parámetros. No
hay loss, no hay backward, no hay gradientes: solo forward, muestreo del próximo
token, y volver a empezar con ese token agregado al final. Corre bastante más
rápido y con menos memoria que el entrenamiento, porque no hace falta guardar
los valores intermedios. En PyTorch se envuelve en `torch.no_grad()` justamente
para eso.

## LayerNorm

Una normalización que agarra los 384 números de una posición, les resta su
promedio y los divide por cuánto se dispersan alrededor de ese promedio (el
desvío estándar). Lo que sale queda centrado en 0 y con una escala comparable
entre capas. Después lo
escala y lo desplaza con dos parámetros que el modelo aprende, para que la capa
pueda recuperar la escala que le sirva. Sin esto, los valores crecen o se
achican de capa en capa hasta que el entrenamiento se vuelve inestable. GPT-2
usa la variante pre-LN: la normalización va antes del attention y antes del
feed-forward, no después.

## learning rate

El tamaño del paso que da el optimizador en cada actualización. Es el
hiperparámetro que más rompe cuando está mal: muy chico y el modelo tarda
eternidades, muy grande y salta por arriba del mínimo y el loss explota. En este
curso no es un número fijo: sube durante el warmup y después baja siguiendo el
cosine schedule.

## logit / logits

Los puntajes crudos que escupe el modelo, uno por cada token del vocabulario. No
son probabilidades: pueden ser negativos, pueden ser enormes, y no suman nada en
particular. Lo único que significan es que un número más alto quiere decir que
el modelo cree más en ese token. Para convertirlos en probabilidades hay que
pasarlos por un softmax.

## loss

Un solo número que mide qué tan equivocado estuvo el modelo, donde más chico es
mejor. Todo el entrenamiento existe para bajar ese número. Acá el loss es cross
entropy, y arranca cerca de `log(V)` en un modelo sin entrenar: con un
vocabulario de 65 caracteres eso da `4.17`. Si tu entrenamiento arranca muy por
encima de ese valor, el bug está en la inicialización y no en el resto.

## loss auxiliar

Un término extra que se suma al loss principal y que castiga el desbalance entre
expertos. Se calcula con la fracción de tokens que recibió cada experto y la
probabilidad promedio que el router le dio, y da un valor mínimo cuando el
reparto es parejo. Se multiplica por un coeficiente chico, típicamente `0.01`,
porque su trabajo es empujar y no dominar: el modelo tiene que seguir
optimizando el loss de lenguaje. Sin esto, el router colapsa en un experto y el
capítulo 13 lo muestra.

## máscara causal

El mecanismo que impide que una posición mire hacia adelante. La posición `t`
puede leer de la `0` hasta la `t`, y nada más allá, porque leer el futuro sería
copiar la respuesta. Se implementa poniendo `-inf` en los scores de attention de
las posiciones prohibidas, antes del softmax: `exp(-inf)` da 0, así que esas
posiciones quedan con peso cero exacto. Gracias a esta máscara, un bloque de 256
tokens produce 256 predicciones en un solo forward.

## merge

En BPE, la regla que dice que el par de tokens `(a, b)` se reemplaza por el
token nuevo `c`. Los merges no son un conjunto sino una lista ordenada, porque
cada uno se construye sobre los anteriores. Un token que representa `aaab` solo
tiene sentido si los merges que armaron `aa` y `aaa` ya se aplicaron. Por eso
`encode` recorre los merges en el mismo orden en que `train` los creó.

## Mixture of Experts / MoE

La arquitectura que reemplaza la capa feed-forward de cada bloque por varias
copias, llamadas expertos, más un router que elige cuáles usa cada token. El
modelo termina con muchos más parámetros totales pero la misma cantidad de
parámetros activos por token, así que gana capacidad sin gastar más cómputo.
Mixtral, DeepSeek y casi todos los modelos grandes de hoy la usan. Es el tema de
la parte 3 del curso, y trae sus propios problemas: colapso del router,
capacidad y descarte de tokens.

## MPS

Sigla de *Metal Performance Shaders*, el backend de PyTorch que corre en la GPU
de una Mac con Apple Silicon. Es el equivalente de CUDA en una máquina de
NVIDIA: mandás los tensores al dispositivo con `.to("mps")` y las operaciones se
ejecutan en la GPU. Los modelos de este curso entrenan en minutos con MPS. En
CPU también andan, con más paciencia.

## muestreo

Elegir el próximo token sorteándolo con las probabilidades que dio el modelo, en
vez de agarrar siempre el más probable. Si el modelo dice 66 por ciento `a` y 24
por ciento `b`, el muestreo saca `a` dos de cada tres veces y `b` una de cada
cuatro. Agarrar siempre el máximo produce texto repetitivo y a veces bucles
infinitos, porque una vez que el modelo se mete en un patrón nada lo saca. La
temperatura y el top-k son las dos perillas que controlan cuánto riesgo toma
este sorteo.

## multi-head attention

Correr varias cabezas de attention en paralelo sobre la misma entrada y
concatenar sus salidas, seguido de una capa lineal que las mezcla. Cada cabeza
trabaja en un subespacio más chico, acá 64 dimensiones de las 384 del modelo. La
versión con loop hace una cabeza por vez y se entiende leyendo; la versión
vectorizada las hace todas juntas con un reordenamiento de dimensiones y es la
que corre rápido. El test del capítulo 5 exige que las dos den el mismo
resultado.

## overfitting

Cuando el modelo mejora sobre los datos de entrenamiento y al mismo tiempo
empeora sobre los datos que nunca vio. Se ve como una tijera en el gráfico: el
loss de entrenamiento sigue bajando y el de validación toca un mínimo y empieza
a subir. Quiere decir que el modelo está memorizando en vez de aprender el
patrón. Ese punto de quiebre es la razón por la que existe el conjunto de
validación.

## parámetro

Cada número que el modelo aprende: los pesos de las matrices, los sesgos, las
filas de la tabla de embeddings. El modelo del curso tiene alrededor de 10
millones, y entrenar es la búsqueda de buenos valores para todos ellos al mismo
tiempo. Cada uno tiene su gradiente y el optimizador lo mueve en cada paso.

## parámetros activos

La cantidad de parámetros que participan realmente en el cálculo de un token, que
en un modelo sparse es mucho menor que el total. Un MoE con 8 expertos y top-2
tiene aproximadamente 4 veces los parámetros de un modelo denso equivalente,
pero cada token usa solo 2 de los 8 expertos, así que la cuenta activa queda
parecida. Este número es el que manda para el costo de cómputo, y también el que
hace que la comparación del capítulo 12 sea justa: el MoE tiene que ganar con
los mismos parámetros activos, no con más.

## perplejidad

La exponencial del loss, o sea `exp(loss)`, y una forma más intuitiva de leer el
mismo número. Se interpreta como entre cuántos tokens duda el modelo en promedio
antes de cada elección. Un loss de `4.17` sobre un vocabulario de 65 caracteres
da una perplejidad de 65, o sea que el modelo no sabe nada y duda entre todos.
Bajar el loss a `1.5` deja una perplejidad de `4.5`: el modelo se debate entre
unas cuatro opciones razonables.

## PyTorch

La biblioteca sobre la que se construye todo el curso. Aporta tres cosas:
tensores con operaciones que corren en GPU, autograd (o sea el backward
calculado solo a partir del forward que escribiste), y los módulos de `torch.nn`
para armar capas. El curso te hace escribir a mano varias operaciones que
PyTorch ya trae, primero para entenderlas, y después usa la versión de la
biblioteca.

## regla de la cadena

La regla que te deja calcular la derivada de una función compuesta multiplicando
las derivadas de cada tramo. Si `y` depende de `x`, y `z` depende de `y`,
entonces `dz/dx = dz/dy * dy/dx`. Pensalo con engranajes: si el segundo gira 3
veces por cada vuelta del primero, y el tercero gira 2 por cada vuelta del
segundo, el tercero gira 6 por cada vuelta del primero. Un modelo es una
composición de cientos de funciones, y el backward es esta regla aplicada capa
por capa desde el loss hasta cada parámetro.

## residual

La conexión que suma la entrada de una sub-capa a su salida: en vez de `y =
f(x)`, el bloque calcula `y = x + f(x)`. Sirve para el gradiente: la suma le da
un camino directo hacia atrás, sin pasar por las multiplicaciones de `f`, así
que en un modelo profundo el gradiente llega vivo hasta las primeras capas. Sin
residuales, apilar 6 bloques hace que el gradiente se apague antes de llegar
abajo. La otra lectura es que cada bloque escribe una corrección sobre lo que
venía, en lugar de reemplazarlo.

## router

La capa lineal chiquita que, en un MoE, mira cada token y le da un puntaje a
cada experto. Esos puntajes pasan por un softmax, se eligen los `k` más altos
con top-k, y el token va a esos expertos. Las probabilidades del softmax se usan
también como pesos para combinar las salidas: si el router le dio 0.7 al experto
3 y 0.3 al 5, la salida es `0.7 * salida_3 + 0.3 * salida_5`. Es la pieza que
hace que el modelo sea sparse, y también la que colapsa si nadie la controla.

## seed / semilla

El número con el que arranca un generador de números al azar. Con la misma
semilla, la secuencia "al azar" sale idéntica en cada corrida, y eso es lo que
te deja comparar dos entrenamientos y saber que la diferencia la causó tu
cambio y no la suerte del sorteo. En este curso aparece como
`torch.Generator().manual_seed(1337)` en los tests, y `get_batch` recibe ese
generador en el argumento `generator`.

## self-attention

Attention donde las consultas, las claves y los valores salen todos de la misma
secuencia. O sea que la secuencia se mira a sí misma: cada token arma su
representación a partir de los otros tokens del mismo texto. Es lo que hace el
transformer en cada bloque. En este curso, además, es causal: cada posición solo
puede mirar hacia atrás.

## self-supervised

El método de entrenamiento donde las etiquetas salen de los propios datos, sin
que nadie las escriba. Acá la etiqueta de cada posición es simplemente el token
de la posición siguiente, así que cualquier texto crudo sirve como dataset. Esa
es la razón por la que los modelos de lenguaje se pueden entrenar con todo
internet: no hace falta que alguien anote nada.

## shape

La tupla con el tamaño de cada dimensión de un tensor. Un tensor de shape
`(32, 256, 384)` tiene 32 bloques, de 256 posiciones, de 384 números cada una.
Es la primera cosa que hay que mirar cuando algo falla, porque la mayoría de los
bugs en este código son un shape que no encaja o una dimensión que se perdió en
una reducción. `x.shape` te lo dice, y un `print` de shapes resuelve más
problemas que leer el código dos veces.

## softmax

Convierte una lista de puntajes cualesquiera en probabilidades: números entre 0
y 1 que suman exactamente 1. Hace tres pasos: le aplica `exp` a cada número,
suma todos los resultados, y divide cada uno por esa suma. El `exp` está para
dos cosas, volver positivo cualquier número (incluso los negativos) y respetar
el orden original. Con puntajes `2.0`, `1.0` y `0.1` salen 66, 24 y 10 por
ciento: fijate que las diferencias se amplifican, y eso es a propósito. En la
práctica siempre se le resta el máximo primero, porque `exp(1000)` desborda y
devuelve `inf`.

## sparse

Un modelo donde cada token activa solo una fracción de los parámetros, en vez de
todos. Es lo que hace un MoE: 8 expertos existen, 2 trabajan por token. La
ventaja es que podés agrandar el modelo sin agrandar el cómputo por token. El
costo es todo lo demás que trae: el router hay que balancearlo, los expertos hay
que darles capacidad, y el dispatch hay que vectorizarlo para que sea rápido de
verdad.

## split

El corte que parte los datos en la parte de entrenamiento y la de validación.
Acá lo hace `train_val_split`, y el corte es contiguo: los primeros 90 por
ciento del texto para entrenar y el último 10 por ciento para validar, sin
mezclar nada. Un split aleatorio no sirve con datos secuenciales, porque las
ventanas de entrenamiento se superponen y quedarían ventanas casi idénticas de
los dos lados del corte.

## temperatura

Un número por el que se dividen los logits antes del softmax, y que controla
cuánto arriesga el muestreo. Con temperatura menor a 1 las diferencias se
agrandan, la distribución se vuelve más filosa y el texto sale más conservador y
repetitivo. Con temperatura mayor a 1 se achatan, y el texto sale más variado y
más propenso al disparate. Con temperatura 1 los logits pasan tal cual, y en el
límite hacia 0 el muestreo se vuelve elegir siempre el máximo.

## tensor

Un arreglo de números con `N` dimensiones, y la estructura de datos con la que
está hecho todo el modelo. Un número suelto tiene 0 dimensiones, un vector 1,
una matriz 2, y de ahí para arriba. En este curso, un batch de tokens es un
tensor de 2 dimensiones `(batch, posición)`, y después de la tabla de embeddings
se vuelve uno de 3 `(batch, posición, canal)`. Es lo mismo que un array de
NumPy, más dos cosas: corre en la GPU y sabe calcular sus propios gradientes.

## Tiny Shakespeare

El corpus del curso: 1115394 caracteres de obras de Shakespeare en un solo
archivo de texto, que bajás con `uv run python scripts/get_data.py`. Tiene 65
caracteres distintos, y de ahí sale el vocabulario de 65 tokens del capítulo 1.
Es chico a propósito: entra entero en memoria y un modelo de 10 millones de
parámetros lo entrena en minutos.

## token

La unidad mínima que el modelo lee y escribe, representada siempre como un
entero. Puede ser un carácter, como en el capítulo 1, o un pedazo de palabra,
como con BPE. El modelo nunca ve texto: ve enteros que después usa como índices
en la tabla de embeddings. Convertir texto a tokens y de vuelta es trabajo del
tokenizador.

## token dropping

Lo que pasa cuando un experto ya llegó a su capacidad y le siguen llegando
tokens: los que sobran no pasan por ese experto. En un MoE con top-2, un token
descartado por uno todavía tiene el otro, así que rara vez se queda sin nada. Si
lo pierde todo, la conexión residual lo salva: el token sale del bloque igual que
entró, sin procesamiento, pero sin agujero. La fracción de tokens descartados es
una métrica que conviene mirar, porque si es alta el factor de capacidad quedó
corto.

## tokenizador

La pieza que convierte texto en una lista de enteros y de vuelta a texto. Tiene
dos métodos, `encode` y `decode`, y son inversos uno del otro. El del capítulo 1
usa un entero por carácter y le alcanzan 65 para Tiny Shakespeare. El del
capítulo 2 usa BPE sobre bytes UTF-8, que es lo que hace GPT-2 con sus 50257
tokens.

## top-k

Quedarse con los `k` valores más altos de una lista y descartar el resto. En un
MoE, es cómo el router elige a qué expertos manda cada token: el modelo del
curso usa top-2 sobre 8 expertos. En el muestreo de texto es otra cosa parecida:
te quedás con los `k` tokens más probables, renormalizás sus probabilidades y
sorteás solo entre ellos, para que la cola larga de tokens absurdos no aparezca
nunca.

## transformer

La arquitectura que usan casi todos los modelos de lenguaje de hoy, y a la que
llega la parte 2 del curso. Se arma apilando bloques iguales, y cada bloque hace
dos cosas: mezclar información entre posiciones con self-attention, y procesar
cada posición por separado con la capa feed-forward. La versión de GPT-2 es
decoder-only, o sea que solo mira hacia atrás con la máscara causal. Lo que la
hizo ganar es que el attention procesa todas las posiciones en paralelo, cosa
que una red recurrente no puede.

## UTF-8

La codificación estándar que convierte texto en bytes. Usa entre 1 y 4 bytes por
carácter: `"a"` es un solo byte, `"ñ"` son dos, un emoji son cuatro. La
propiedad que le importa a BPE es la cobertura: cualquier texto de cualquier
idioma se representa con bytes del 0 al 255, así que un vocabulario base de 256
tokens no deja nada afuera y nunca hace falta un token desconocido.

## vocabulario

El conjunto completo de tokens que el modelo conoce, y su tamaño se escribe `V`.
Es la cantidad de filas de la tabla de embeddings y la cantidad de logits que
sale por cada posición. El tokenizador de caracteres del capítulo 1 tiene
`V = 65`, y GPT-2 tiene `V = 50257`. Un vocabulario chico da secuencias largas y
tablas chicas; uno grande da lo contrario, y ese es el trade-off.

## warmup

Los primeros pasos del entrenamiento, en los que el learning rate sube desde
casi cero hasta su valor máximo en lugar de arrancar arriba. Existe porque al
principio los parámetros son ruido y los gradientes son enormes y poco
confiables: un paso grande en esa situación manda al modelo a cualquier lado. Se
usan unos pocos cientos de pasos, y después empieza a bajar por el cosine
schedule.

## weight tying

Usar la misma matriz para la tabla de embeddings de entrada y para la capa que
produce los logits de salida. Se puede porque las dos relacionan lo mismo,
tokens con vectores de dimensión `d`, solo que en direcciones opuestas. Con
`V = 50257` y `d = 768`, compartirla ahorra 38 millones de parámetros, que en
GPT-2 no es poco. GPT-2 lo hace, así que tu modelo lo tiene que hacer también
para que los logits del capítulo 7 den iguales.

## z-loss

Un término extra que castiga que los logits del router se vayan haciendo grandes
en valor absoluto. Se calcula sobre el log-sum-exp de esos logits, que es
justamente el denominador del softmax, y lo empuja hacia valores chicos. Sirve
para la estabilidad numérica: logits grandes producen exponenciales grandes, y
en float16 eso termina en `inf` y después en `nan`. El paper ST-MoE lo introdujo
con un coeficiente de `0.001`, y es distinto del loss auxiliar: uno cuida el
balance entre expertos, el otro cuida los números.
