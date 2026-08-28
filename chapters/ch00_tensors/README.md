# Capítulo 0: tensores, softmax y gradientes

## Por qué este capítulo va primero

### Antes que nada: qué es un tensor

Un tensor es una caja de números, todos del mismo tipo, ordenados en una
grilla. Nada más que eso. Lo que cambia de un tensor a otro es cuántas
dimensiones tiene la grilla:

```
3.7                          0 dimensiones. Un solo número. Se llama escalar.
[3.7, 1.2, 0.5]              1 dimensión. Una fila de 3 números.
[[3.7, 1.2, 0.5],            2 dimensiones. Una tabla de 2 filas por 3 columnas.
 [0.1, 8.0, 2.2]]
```

Si seguís apilando, seguís sumando dimensiones. Tres tablas del mismo tamaño,
una atrás de la otra, son un tensor de 3 dimensiones. Un tensor de 4
dimensiones son varias de esas pilas. La palabra "tensor" es simplemente el
nombre general: sirve para cualquier cantidad de dimensiones, incluso cero.

A la lista de tamaños de cada dimensión se le dice **shape**. La tabla de arriba
tiene shape `(2, 3)`: dos filas, tres columnas. En Python lo mirás así:

```python
t = torch.tensor([[3.7, 1.2, 0.5], [0.1, 8.0, 2.2]])
t.shape        # torch.Size([2, 3])
```

Un ejemplo del curso, para que el shape signifique algo. Tomá un caso chico,
del tamaño que podés seguir a mano: 32 fragmentos de texto de 8 tokens cada
uno. El entrenamiento del capítulo 8 usa fragmentos de 256 tokens, que es el
contexto del modelo, pero la lectura del shape es idéntica. Un **token**
es cada pedacito de texto que el modelo trata como una unidad indivisible: en
los primeros capítulos es un solo carácter, y desde el capítulo 2 pasa a ser un
pedazo de palabra. A ese grupo de 32 fragmentos que entran juntos en una misma
pasada se le dice **batch**, y entran juntos porque la placa hace las 32
cuentas en paralelo casi al mismo costo que una.

El modelo devuelve un puntaje para cada uno de los 65 caracteres posibles, en
cada posición de cada fragmento. Ese resultado es un tensor de shape
`(32, 8, 65)`. Se lee: 32 fragmentos, 8 posiciones por fragmento, 65 puntajes
por posición. Son 16.640 números en una sola caja.

Todo el modelo que vas a escribir es una cadena de operaciones sobre tensores,
y la mayor parte de los bugs que te vas a comer no son errores de matemática:
son shapes que no encajan, o peor, shapes que encajan por casualidad y hacen la
cuenta equivocada sin avisar. Por eso el curso te va a hacer mirar el shape
todo el tiempo.

### Las tres operaciones de este capítulo

Hay tres cosas que aparecen en todos los capítulos que siguen.

Un **softmax** convierte puntajes crudos en probabilidades. Se usa dos veces en
el modelo: para decidir cuánta atención le presta cada token a los demás, y
para decir con qué probabilidad sigue cada carácter.

Un loss de **cross entropy** mide qué tan equivocadas están esas
probabilidades, con un solo número.

La **regla de la cadena** agarra ese número y lo lleva de vuelta hasta cada
parámetro del modelo, para decirle a cada uno hacia dónde moverse.

PyTorch tiene las tres, y de acá en adelante el curso usa la versión de
PyTorch. Las escribís una sola vez, acá, porque un bug en un capítulo
posterior se encuentra mucho más rápido cuando ya sabés qué hace cada una de
estas operaciones.

## 1. Qué es un softmax y por qué lo necesitás

El modelo termina su trabajo escupiendo un número por cada token del
vocabulario. Con 65 caracteres posibles, son 65 números. Algo así:

```
'a': 2.0    'b': 1.0    'c': 0.1    ...
```

Esos números se llaman **logits**. No son probabilidades: son puntajes crudos,
sin escala fija. Pueden ser negativos, pueden ser gigantes, y no suman nada en
particular. Lo único que significan es que un número más alto es "el modelo
cree más en este token".

Pero vos necesitás probabilidades. Necesitás poder decir "hay 66% de chance de
que siga una `a`", porque para entrenar hay que medir qué tan equivocado
estuvo, y para generar texto hay que sortear el próximo token. Un puntaje de
`2.0` no te sirve para ninguna de las dos cosas.

**Un softmax es la máquina que convierte esos puntajes en probabilidades.**
Entra una lista de números cualesquiera, sale una lista de números entre 0 y 1
que suman exactamente 1.

### Por qué no alcanza con dividir por la suma

La idea obvia sería dividir cada número por la suma de todos. No funciona, por
dos motivos. Si hay negativos te salen probabilidades negativas, que no
existen. Y si los números suman cero, dividís por cero.

Por eso primero se pasa todo por `exp()`. La exponencial tiene dos propiedades
que la hacen la elección correcta: convierte cualquier número en uno positivo
(`exp(-5)` da `0.0067`, chiquito pero positivo), y respeta el orden, o sea que
si `a > b` entonces `exp(a) > exp(b)`. Nadie se adelanta en la fila.

Con los puntajes de arriba:

```
paso 1, exponencial:    exp(2.0)=7.39   exp(1.0)=2.72   exp(0.1)=1.11
paso 2, sumar:          7.39 + 2.72 + 1.11 = 11.22
paso 3, dividir:        7.39/11.22=0.66  2.72/11.22=0.24  1.11/11.22=0.10
```

Y ahí tenés: 66%, 24%, 10%. Suman 1.

Fijate que la diferencia se **amplificó**. En puntajes, `2.0` era el doble de
`1.0`. En probabilidades, 66% es casi el triple de 24%. Eso lo hace la
exponencial, y es a propósito: el modelo se compromete con su favorito en vez
de repartir parejo.

Escrita como fórmula, esa cuenta de tres pasos es:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Lee así: la probabilidad del elemento `i` es su exponencial dividida por la
suma de todas las exponenciales. Es literalmente lo que hicimos recién.

### El problema: esa fórmula, pasada a código, explota

Pasar la fórmula directo a código falla. Con `z_i = 1000`, `exp(1000)` es más
grande que el float más grande que existe, así que el resultado es `inf`.
Después `inf / inf` da `nan`, y ese `nan` se propaga por todo el modelo: toda
cuenta que toque un `nan` devuelve `nan`, así que un solo valor podrido te
arruina el entrenamiento entero y no te dice dónde empezó.

No es un caso raro de laboratorio. Los logits crecen solos a medida que el
modelo se vuelve más seguro de sus predicciones.

La corrección usa una propiedad del softmax: si le restás una constante `c` a
cada elemento de `z`, el resultado no cambia. Acá está el porqué, en un renglón:

```
exp(z_i - c) / sum_j exp(z_j - c) = [exp(z_i) exp(-c)] / [exp(-c) sum_j exp(z_j)]
```

Eso sale de que `exp(a - b) = exp(a) * exp(-b)`. El factor `exp(-c)` queda en el
numerador y también en el denominador, así que se cancela, y lo que queda es el
softmax original. O sea que restar la constante es gratis: cambia los números
intermedios, no el resultado.

La elección útil es `c = max(z)`, el valor más grande de la fila. Después de la
resta, el elemento más grande queda en `0`, y todos los demás quedan negativos.
Como `exp` de cero es `1` y `exp` de un negativo es menor que `1`, todas las
exponenciales caen entre `0` y `1`. No hay overflow posible, o sea que ningún
resultado se pasa del número más grande que un float puede guardar.

Con la fila del test, `[1000, 1000, 1001]`, el máximo es `1001`. La fila
desplazada es `[-1, -1, 0]`, y `exp` de eso da `[0.37, 0.37, 1.0]`. Números
domesticados, mismo resultado.

Esto no es un detalle de estilo. GPT-2 divide los puntajes de attention por la
raíz cuadrada de la dimensión de la cabeza por una razón parecida, y el
capítulo 4 la explica.

**Shapes.** Tu función `softmax_rows` recibe un tensor 2-D con shape `(N, C)` y
devuelve un tensor con el mismo shape. Cada fila tiene que sumar `1`.

Acá aparece el **broadcast**, que es la regla de PyTorch para operar dos
tensores de shape distinto: si una dimensión mide `1` de un lado, ese valor se
repite tantas veces como haga falta para igualar al otro. Restar un tensor
`(N, 1)` a uno `(N, C)` funciona porque el único valor de cada fila se repite en
las `C` columnas, que es exactamente lo que querés.

Por eso mirá bien el argumento `keepdim` de `max` y de `sum`. Una reducción sin
`keepdim` saca la dimensión y te deja shape `(N,)` en vez de `(N, 1)`, y ahí el
broadcast alinea las dimensiones desde la derecha y termina repitiendo sobre la
dimensión equivocada, o directamente falla.

## 2. Qué es el cross entropy y por qué el logaritmo

### Qué querés medir

Ya tenés las probabilidades. Ahora aparece el texto real y el carácter que
seguía era una `a`. ¿Estuvo bien el modelo?

Para entrenar necesitás una respuesta que sea **un número**, no un sí o un no.
Un número que sea chico cuando el modelo acertó y grande cuando erró, y que se
mueva de a poco: si el modelo mejora apenas, el número tiene que bajar apenas.
Ese número se llama **loss**.

**El cross entropy mide el error mirando una sola cosa: qué probabilidad le
dio el modelo al token que efectivamente apareció.** El resto de la
distribución no se mira. Si le dio mucha, el loss es chico. Si le dio poca, el
loss es grande.

### Por qué no sirve contar aciertos

Lo obvio sería medir el porcentaje de veces que el modelo le pegó al carácter
más probable. No sirve, y el motivo es importante.

Pensá dos modelos que aciertan igual de seguido. Uno le da 0.51 al carácter
correcto, el otro le da 0.99. Contando aciertos valen lo mismo, pero el segundo
es muchísimo mejor y vos querés premiarlo. Peor todavía: si el modelo mejora de
0.20 a 0.45 sin llegar a ganar, el conteo de aciertos no se mueve ni un poco.
Un número que no se mueve no te dice hacia dónde ir, y todo el entrenamiento
consiste justamente en saber hacia dónde ir.

Entonces usemos la probabilidad misma. Casi. Le falta una vuelta.

### Qué hace el logaritmo con un número entre 0 y 1

El logaritmo natural, `log`, contesta esta pregunta: a qué exponente hay que
elevar `e` (que vale más o menos `2.718`) para obtener este número. Es la
operación inversa de `exp`.

Para números entre 0 y 1 la respuesta siempre es negativa, y se va a lo hondo
muy rápido:

```
log(1.0)    =  0
log(0.5)    = -0.69
log(0.1)    = -2.30
log(0.01)   = -4.61
log(0.001)  = -6.91
log(0.0)    = -infinito
```

Mirá el salto. Entre `1.0` y `0.5` el logaritmo se movió `0.69`. Entre `0.01` y
`0.001`, o sea entre dos probabilidades que las dos son "casi nada", se movió
`2.30`, más del triple. **El logaritmo castiga muchísimo más estar seguro y
equivocado que estar dudando.** Eso es exactamente lo que querés de un loss:
que un modelo que dice "es imposible que siga una `a`" y después aparece una
`a` la pague carísimo.

Hay un segundo motivo, más práctico. La probabilidad de una frase entera es el
producto de la probabilidad de cada token, y multiplicar mil números menores
que 1 da un valor tan chiquito que en floats termina siendo `0`. Los logaritmos
en cambio se suman, porque `log(a*b) = log(a) + log(b)`, y sumar mil números
del orden de `-3` no rompe nada.

### El signo menos

El logaritmo de una probabilidad siempre es negativo o cero. Un loss negativo
es incómodo: querés que "más grande" quiera decir "peor" y que el mínimo sea
`0`. El menos adelante da vuelta el signo y listo. Nada más profundo que eso.

```
loss = -log(p_del_token_correcto)
```

### El ejemplo numérico

Seguimos con las probabilidades del softmax de la sección anterior:

```
'a': 0.66    'b': 0.24    'c': 0.10
```

Caso 1, el texto real seguía con `a`. El modelo le había dado `0.66`.

```
loss = -log(0.66) = 0.42
```

Caso 2, el texto real seguía con `c`. El modelo le había dado `0.10`.

```
loss = -log(0.10) = 2.30
```

Cinco veces más loss por el mismo error de un solo carácter. Y si el modelo le
hubiese dado `0.001` a la `c`, el loss sería `6.91`.

### Cómo se lee un loss

Un loss de `6.9` quiere decir que el modelo le dio al token correcto una
probabilidad de más o menos `0.001`, porque `exp(-6.9)` da eso. Esa es la
lectura literal.

La lectura útil es otra. Si un modelo reparte la probabilidad en partes iguales
entre `k` opciones, cada una se lleva `1/k` y el loss da `log(k)`. Dado un loss
`L`, entonces, `exp(L)` te dice entre cuántas opciones parejas está dudando el
modelo:

```
loss 0.00  ->  1 opción.     Certeza total y correcta.
loss 0.69  ->  2 opciones.   Está entre dos, en partes iguales.
loss 2.30  ->  10 opciones.
loss 4.17  ->  65 opciones.  No sabe nada: el vocabulario entero.
loss 6.90  ->  1000 opciones.
```

Con esa tabla el loss deja de ser un número abstracto. Un modelo de caracteres
que arranca en `4.17` está adivinando, y está bien que arranque ahí.

### La fórmula

El modelo da un logit a cada uno de los `C` tokens del vocabulario. El token
correcto tiene índice `t`. Entonces:

```
loss = -log(softmax(z)_t)
```

Y como entrenás con un batch de `N` filas a la vez, el loss del batch es el
promedio de las `N` filas. Tu función `cross_entropy` toma logits con shape
`(N, C)` y targets con shape `(N,)`, y devuelve un escalar, o sea un tensor sin
dimensiones, con shape `()`. El test lo verifica.

### Acá también hay que cuidar la estabilidad

Escribir `log(softmax(z))` tal cual funciona en el pizarrón y falla en la
máquina. Mirá lo que pasa con la fila del test, `[0.0, 0.0, 800.0]`, cuando el
token correcto es el primero. Si no restás el máximo, `exp(800)` da `inf`. Y si
lo restás, el softmax devuelve `[0.0, 0.0, 1.0]`, porque las dos primeras
probabilidades son tan chicas que en float redondean a cero. Después
`log(0)` da `-infinito`, y el loss sale `inf`. El modelo estaba equivocado, sí,
pero el loss tiene que ser un número finito para que el gradiente sirva de algo.

El álgebra te saca ese ida y vuelta de encima. Arrancá de la definición y
aplicá dos propiedades del logaritmo, `log(a/b) = log(a) - log(b)` y
`log(exp(u)) = u`:

```
log_softmax(z)_i = log( exp(z_i - m) / sum_j exp(z_j - m) )        con m = max(z)
                 = (z_i - m) - log( sum_j exp(z_j - m) )
```

El `z_i - m` de adelante es una resta común, sin exponencial ni logaritmo en el
medio, así que nunca se va a cero. Con la fila de arriba te da `0 - 800 = -800`
menos un término chiquito: un loss de `800`, que es enorme y correcto. Por eso
el docstring de `cross_entropy` te pide calcular el log-softmax directo, sin
llamar a `log()` después de `exp()`.

**Un número para acordarse.** Un modelo sin entrenar le da más o menos la
misma probabilidad a cada token, así que el loss ronda `log(C)`. Con un
vocabulario de 65 caracteres ese valor da cerca de `4.17`. El capítulo 3 usa
este número como primer chequeo del primer modelo, y el test
`test_cross_entropy_of_a_uniform_model` lo verifica acá mismo. Un
entrenamiento que arranca muy por encima de `log(C)` tiene un bug en la
inicialización.

## 3. Qué es un gradiente, y el backward de una capa lineal

Esta es la sección larga, porque acá está la idea que hace funcionar todo lo
demás.

### El problema

Ya tenés un número que dice qué tan mal está el modelo. Ahora hay que
arreglarlo, y arreglarlo quiere decir cambiar los pesos: los millones de
números que el modelo tiene adentro. La pregunta es brutalmente concreta:
**para cada peso, ¿lo subo o lo bajo, y cuánto?**

Probar no es opción. Mover un peso, correr el modelo entero y ver si el loss
bajó cuesta una pasada completa. Con millones de pesos, y repitiendo miles de
veces, la cuenta no cierra ni de casualidad.

Lo que sí se puede es calcularlo. Y para eso hace falta una derivada.

### Qué es una derivada

Una derivada contesta una sola pregunta: **si muevo la entrada un poquito, ¿cuánto
se mueve la salida?**

Tomá `f(w) = w * w` y pará en `w = 3`, donde `f` vale `9`. Movete un poquito, a
`w = 3.01`:

```
f(3.00) = 9.0000
f(3.01) = 9.0601
cambió 0.0601 la salida, por 0.01 que moviste la entrada
0.0601 / 0.01 = 6.01
```

Ese `6.01` es la respuesta: cerca de `w = 3`, la salida cambia unas 6 veces más
rápido que la entrada. Si achicás el paso, el número se acerca cada vez más a
`6` exacto. Eso es la derivada de `w * w` en `w = 3`, y se escribe `df/dw = 6`.

Dos lecturas que te sirven de acá en adelante. El **signo** te dice para qué
lado: positivo quiere decir "si subo `w`, la salida sube". El **tamaño** te dice
cuánto le importa: un `6` mueve la salida seis veces más que un `1`.

### Qué es una derivada parcial

Arriba había una sola entrada. En un modelo hay millones. Una derivada parcial
resuelve eso de la manera más simple posible: **movés una sola entrada y dejás
todas las demás quietas.** Nada más. El símbolo cambia de `d` a `∂` para
avisar que las otras están congeladas, pero la idea es idéntica.

Con dos entradas, `f(a, b) = a * b + b`, parados en `a = 2`, `b = 5`, donde `f`
vale `15`:

```
movés a:  f(2.01, 5.00) = 10.05 + 5.00 = 15.05    subió 0.05 / 0.01  ->  ∂f/∂a = 5
movés b:  f(2.00, 5.01) = 10.02 + 5.01 = 15.03    subió 0.03 / 0.01  ->  ∂f/∂b = 3
```

Fijate que las dos derivadas son distintas. `b` mueve el resultado por dos
caminos a la vez, porque aparece en los dos términos, y `a` por uno solo. En un
modelo pasa lo mismo a lo grande: cada peso influye en el loss de una forma
distinta, y por eso cada uno necesita su propio número.

### Qué es el gradiente

**El gradiente es simplemente todas las derivadas parciales juntas, una por
entrada, guardadas en un tensor con la misma forma que las entradas.** En el
ejemplo de arriba, el gradiente de `f` es `[5, 3]`.

Y tiene una propiedad que lo vuelve la pieza central del entrenamiento: ese
vector apunta en la dirección en la que la función **crece** más rápido. Si te
parás en un punto y das un paso en la dirección del gradiente, subís lo más
posible.

Pero vos querés bajar el loss, no subirlo. Así que das el paso al revés, y ahí
tenés el entrenamiento entero en un renglón:

```
peso = peso - learning_rate * dL/dpeso
```

El menos es el "para el otro lado". El `learning_rate` es un número chico que
controla el tamaño del paso, porque el gradiente vale cerca del punto donde lo
calculaste y no a diez kilómetros.

Sobre la notación: `dL/dw` se lee "cuánto cambia el loss `L` por cada unidad
que muevas `w`". En el código lo vas a ver como `grad_w`, y el gradiente del
loss respecto de la salida de la capa te llega en el argumento `grad_out`, que
es `dL/dy`.

### La regla de la cadena, y por qué existe

Acá está el problema real. El loss no toca los pesos de la primera capa. Los
toca a través de la segunda, que pasa por la tercera, y así hasta el final. Un
peso allá abajo influye en el loss por un camino largo. ¿Cómo calculás esa
derivada sin volver a recorrer el camino entero por cada peso?

Pensá en engranajes. Si una vuelta de la manija hace girar tres vueltas al
engranaje del medio, y una vuelta del engranaje del medio hace girar dos
vueltas a la rueda final, entonces una vuelta de la manija son seis vueltas de
la rueda. Multiplicás las relaciones. La regla de la cadena dice exactamente
eso:

```
dL/dx = dL/dy * dy/dx
```

En palabras: cuánto le importa `x` al loss es igual a cuánto le importa `y` al
loss, multiplicado por cuánto mueve `x` a `y`.

Con números. Suponé `y = 3x` y `L = 2y`, parados en `x = 1`:

```
dy/dx = 3          mover x en 0.01 mueve y en 0.03
dL/dy = 2          mover y en 0.03 mueve L en 0.06
dL/dx = 2 * 3 = 6  y efectivamente L pasó de 6 a 6.06
```

**Y acá está el porqué de todo el diseño.** Cada capa necesita saber una sola
cosa del mundo exterior: `dL/dy`, cuánto le importa su propia salida al loss.
Con eso, y con lo que la capa sabe de sí misma, calcula los gradientes de sus
pesos y además calcula el `dL/dy` de la capa anterior, y se lo pasa. La última
capa arranca la cadena, y el número viaja hacia atrás capa por capa hasta la
primera. De ahí el nombre **backward**. Sin esta regla habría que recalcular
todo el camino para cada peso; con ella, una sola pasada hacia atrás alcanza
para todos.

### Ahora sí, la capa lineal

Una capa lineal es `y = x @ W + b`. Es la operación más común del modelo: cada
salida es una suma ponderada de todas las entradas, más un corrimiento fijo.

Bajemos a un caso mínimo que podés seguir a mano. Una sola fila, dos entradas,
dos salidas:

```
x  = [1.0, 2.0]              shape (1, 2)
W  = [[2.0, 3.0],            shape (2, 2)
      [1.0, 4.0]]
```

Escrito sin matrices, para ver quién multiplica a quién:

```
y1 = x1*W11 + x2*W21 + b1  =  1*2 + 2*1 + b1
y2 = x1*W12 + x2*W22 + b2  =  1*3 + 2*4 + b2
```

Y suponé que la cadena ya te trajo, desde arriba, el gradiente del loss
respecto de la salida:

```
dL/dy = [0.5, -2.0]
```

O sea: si `y1` sube un poquito, el loss sube un poco; si `y2` sube un poquito,
el loss baja bastante.

**Gradiente respecto de `x`.** `x1` aparece en `y1` multiplicado por `W11`, y en
`y2` multiplicado por `W12`. Toca el loss por los dos caminos, así que aplicás
la regla de la cadena en cada uno y sumás:

```
dL/dx1 = dL/dy1 * W11 + dL/dy2 * W12 = 0.5*2.0 + (-2.0)*3.0 = -5.0
dL/dx2 = dL/dy1 * W21 + dL/dy2 * W22 = 0.5*1.0 + (-2.0)*4.0 = -7.5
```

Eso, escrito con matrices, es `dL/dy @ W.T`. El `.T` es la **transpuesta**:
la misma matriz con las filas puestas como columnas, o sea que el elemento de
la fila `i` y la columna `j` pasa a la fila `j` y la columna `i`. Comprobalo:
`W` es `[[2,3],[1,4]]`, así que `W.T` es `[[2,1],[3,4]]`, y
`[0.5, -2.0] @ [[2,1],[3,4]]` da `[-5.0, -7.5]`. Los mismos dos números.

**Gradiente respecto de `W`.** `W11` aparece en un solo lugar, multiplicando a
`x1` dentro de `y1`. Entonces `dy1/dW11 = x1`, y por la cadena:

```
dL/dW11 = dL/dy1 * x1 = 0.5 * 1.0 =  0.5
dL/dW12 = dL/dy2 * x1 = -2.0 * 1.0 = -2.0
dL/dW21 = dL/dy1 * x2 = 0.5 * 2.0 =  1.0
dL/dW22 = dL/dy2 * x2 = -2.0 * 2.0 = -4.0
```

Cada elemento del gradiente es una entrada por un gradiente de salida. Esa
tabla de todos contra todos es `x.T @ dL/dy`, que da `[[0.5, -2.0], [1.0, -4.0]]`.

Y fijate lo que significa: si una entrada valía `0`, su peso no cambia. Tiene
sentido, porque un peso que multiplica a cero no participó del resultado.

**Gradiente respecto de `b`.** `b1` se suma a `y1` tal cual, sin multiplicar
nada, así que `dy1/db1 = 1` y el gradiente pasa entero: `dL/db = dL/dy`. Con
una sola fila termina ahí. Pero cuando el batch tiene `N` filas, **el mismo `b`
se suma a todas**, así que las `N` filas le mandan su error al mismo número, y
hay que sumarlas.

### Las tres fórmulas

```
dL/dx = dL/dy @ W.T
dL/dW = x.T @ dL/dy
dL/db = dL/dy sumado sobre la dimensión del batch
```

Con mirar los shapes alcanza para acordarse. Con `x` de shape `(N, in)`, `W`
de shape `(in, out)` y `dL/dy` de shape `(N, out)`, hay un solo orden de cada
producto que da el shape correcto.

Tu función `linear_backward` devuelve los tres gradientes como una tupla
`(grad_x, grad_w, grad_b)`, con los shapes de `x`, `w` y `b`. Ojo con
`grad_b`: tiene shape `(out_features,)`, no `(1, out_features)`. El test
`test_linear_backward_with_one_row` corre con `N = 1` justamente para eso, o
sea que la suma sobre el batch tiene que estar aunque haya una sola fila.

La función no recibe `b`, y no es un olvido: el gradiente de una suma no
depende de los valores que se suman. `dL/db` sale de `dL/dy` y nada más.

El test compara tus tres tensores contra los que calcula el autograd de
PyTorch, que corre con `y.backward(grad_out)`. El autograd es el motor que
anota cada operación del forward y después aplica la regla de la cadena solo,
sin que vos escribas ninguna derivada. Si los números dan iguales, tu
derivación está bien.

## Tu tarea

1. Abrí `exercise.py`.
2. Escribí las tres funciones. No uses `torch.softmax`,
   `torch.nn.functional.softmax`, `torch.log_softmax`,
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
