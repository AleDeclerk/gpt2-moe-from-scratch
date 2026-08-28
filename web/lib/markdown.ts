import rehypePrettyCode from "rehype-pretty-code";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";
import { rehypeGlossaryLinks } from "./glossaryLinks";

/** Pasar a HTML el markdown de un capítulo, en tiempo de build.
 *
 * La fuente es un archivo de este mismo repositorio, así que el contenido es
 * confiable.
 *
 * Si se le pasan las formas del glosario, enlaza la primera aparición de cada
 * término. La página del glosario no las pasa, para no enlazarse a sí misma.
 */
export async function renderMarkdown(
  source: string,
  formasDelGlosario: { forma: string; slug: string }[] = [],
): Promise<string> {
  let pipeline = unified().use(remarkParse).use(remarkGfm).use(remarkRehype).use(rehypeSlug);

  if (formasDelGlosario.length > 0) {
    pipeline = pipeline.use(rehypeGlossaryLinks, { formas: formasDelGlosario });
  }

  const file = await pipeline
    .use(rehypePrettyCode, {
      theme: "vitesse-dark",
      keepBackground: false,
      defaultLang: "text",
    })
    .use(rehypeStringify)
    .process(source);
  return String(file);
}

/** Sacar el primer título, porque la página ya lo muestra en su encabezado. */
export function stripFirstHeading(source: string): string {
  return source.replace(/^#\s+.*\n+/, "");
}
