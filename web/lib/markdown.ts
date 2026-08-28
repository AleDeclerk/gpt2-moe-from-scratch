import rehypePrettyCode from "rehype-pretty-code";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

/** Renderiza a HTML el markdown de un capítulo, en tiempo de build.
 *
 * La fuente es un archivo de este repo, así que el contenido es confiable.
 */
export async function renderMarkdown(source: string): Promise<string> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype)
    .use(rehypeSlug)
    .use(rehypePrettyCode, {
      theme: "vitesse-dark",
      keepBackground: false,
      defaultLang: "text",
    })
    .use(rehypeStringify)
    .process(source);
  return String(file);
}

/** Saca el primer heading, porque la página ya muestra el título en su header. */
export function stripFirstHeading(source: string): string {
  return source.replace(/^#\s+.*\n+/, "");
}
