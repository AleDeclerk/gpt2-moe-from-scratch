import type { Element, Root, Text } from "hast";
import { visitParents } from "unist-util-visit-parents";

/** Etiquetas donde NO hay que linkear nada.
 *
 * El código se deja en paz porque un link adentro de un bloque de código lo
 * rompe visualmente. Los títulos también, porque ya son anclas.
 */
const SIN_LINKS = new Set(["code", "pre", "a", "h1", "h2", "h3", "h4", "h5", "h6"]);

/** Escapar los caracteres que tienen significado en una expresión regular. */
function escapar(texto: string): string {
  return texto.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Enlazar al glosario la PRIMERA aparición de cada término en la página.
 *
 * Solo la primera: linkear las veinte apariciones de "softmax" en un capítulo
 * llenaría el texto de azul y no ayudaría a nadie.
 */
export function rehypeGlossaryLinks(options: {
  formas: { forma: string; slug: string }[];
}) {
  const { formas } = options;

  return (tree: Root) => {
    const yaEnlazados = new Set<string>();

    visitParents(tree, "text", (node: Text, ancestros) => {
      const dentroDeZonaProhibida = ancestros.some(
        (a) => a.type === "element" && SIN_LINKS.has((a as Element).tagName),
      );
      if (dentroDeZonaProhibida) return;

      const padre = ancestros[ancestros.length - 1] as Element | Root;
      if (!padre || !("children" in padre)) return;

      for (const { forma, slug } of formas) {
        if (yaEnlazados.has(slug)) continue;

        // Los límites de palabra de JavaScript no entienden los acentos, así
        // que el borde se define a mano con las clases unicode de letra y
        // número. Sin esto, "batch" se encontraría dentro de "batches".
        const patron = new RegExp(
          `(?<![\\p{L}\\p{N}_-])(${escapar(forma)})(?![\\p{L}\\p{N}_-])`,
          "iu",
        );
        const encontrado = patron.exec(node.value);
        if (!encontrado) continue;

        const inicio = encontrado.index;
        const fin = inicio + encontrado[1].length;
        const nuevos: (Text | Element)[] = [];

        if (inicio > 0) nuevos.push({ type: "text", value: node.value.slice(0, inicio) });
        nuevos.push({
          type: "element",
          tagName: "a",
          properties: {
            href: `/glosario#${slug}`,
            className: ["termino"],
            title: `Ver "${forma}" en el glosario`,
          },
          children: [{ type: "text", value: encontrado[1] }],
        });
        if (fin < node.value.length) {
          nuevos.push({ type: "text", value: node.value.slice(fin) });
        }

        const posicion = padre.children.indexOf(node as never);
        padre.children.splice(posicion, 1, ...(nuevos as never[]));
        yaEnlazados.add(slug);
        // Un solo reemplazo por nodo de texto: el nodo ya no existe, y seguir
        // buscando adentro de él daría un índice que no corresponde a nada.
        return;
      }
    });
  };
}
