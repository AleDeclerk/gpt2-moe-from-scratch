# Diseño: gpt2-moe-from-scratch

Fecha: 2026-08-28. Estado: aprobado, en construcción.

## Objetivo

Un repositorio que enseña GPT-2 y Mixture of Experts, con código que escribe
el lector. El lector es el dueño del repositorio, y lo que se busca es
entender a fondo cada parte, no llegar rápido a un resultado.

## Decisiones

| Decisión | Qué se eligió | Por qué |
|---|---|---|
| Formato de los ejercicios | Módulos de Python con pytest | El verde no es una opinión. Un notebook no tiene una condición de pasar clara, y git maneja mucho mejor un archivo `.py` |
| Punto de partida | Todo desde cero, tokenizador incluido | El pedido dice "repasar los conceptos". Una base ya escrita saca justo lo que hay que repasar |
| Profundidad del MoE | Top-k, balanceo de carga, capacidad, diagnósticos | Un router solo colapsa en un único experto. Las otras partes son la respuesta a ese problema |
| Acoplamiento entre capítulos | Promoción | Ver más abajo |
| Idioma | Inglés, con las reglas de ASD-STE100 | Regla global para todo lo que va a GitHub |

## El modelo de promoción

Para la relación entre capítulos había tres modelos posibles:

1. **Autocontenido.** Cada capítulo guarda una copia del código de los
   capítulos anteriores. No hay acoplamiento, pero el lector escribe GPT-2
   doce veces.
2. **Paquete compartido, ya escrito.** El paquete `gpt2moe/` viene completo y
   los capítulos importan de ahí. No hay duplicación, pero la solución se ve
   desde el día uno.
3. **Promoción (la elegida).** El lector escribe `exercise.py`. Cuando los
   tests pasan, `scripts/promote.py` copia el archivo dentro de `gpt2moe/`. El
   capítulo siguiente importa del paquete, así que importa el código del
   propio lector.

El tercer modelo no duplica nada ni adelanta la solución, y el paquete deja
ver cómo avanza el progreso. El costo es un comando por capítulo.

`--from-solution` promueve el código de referencia sin correr los tests. Es la
salida para un capítulo donde el lector se traba.

## Cómo se elige el target de los tests

La variable de entorno `MOE_TARGET` decide entre `exercise` y `solution`. El
fixture `target` de `conftest.py` carga el archivo por ruta, con un nombre de
módulo único para cada capítulo. El nombre único hace falta porque en todos
los capítulos los archivos se llaman igual.

Es el mismo patrón que `PREP_TARGET` en `python-intermediate-prep`.

## Escala del modelo

6 capas, 6 cabezas, 384 dimensiones, un contexto de 256 tokens, cerca de 10
millones de parámetros. Un entrenamiento tarda minutos en MPS. La variante MoE
del capítulo 12 usa 8 expertos con top-2. Eso da unas 4 veces los parámetros
totales, con casi la misma cantidad de parámetros activos.

## Criterios de aceptación

| Capítulo | Condición |
|---|---|
| ch05 | El attention vectorizado coincide con la versión con loop, `allclose` |
| ch07 | El modelo con los pesos reales de `gpt2` da los logits de referencia, atol 1e-4 |
| ch08 | El modelo denso baja del loss de validación objetivo |
| ch11 | El MoE vectorizado coincide con el MoE naive, `allclose` |
| ch12 | El modelo MoE llega a un loss de validación más bajo que el modelo denso, con los mismos parámetros activos |

El criterio del capítulo 7 es el más fuerte. Un modelo que, a partir de los
pesos reales, produce los logits del GPT-2 real es GPT-2, no algo parecido.

## Estado

Listo: el andamiaje, ch00, ch01, ch02. 39 tests.
Sigue: de ch03 a ch08, y después la parte de MoE.
