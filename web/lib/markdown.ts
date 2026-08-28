import rehypePrettyCode from "rehype-pretty-code";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

/** Render the markdown of a chapter to HTML, at build time.
 *
 * The source is a file of this repository, so the content is trusted.
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

/** Remove the first heading, because the page shows the title in its header. */
export function stripFirstHeading(source: string): string {
  return source.replace(/^#\s+.*\n+/, "");
}
