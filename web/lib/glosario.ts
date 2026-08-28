import fs from "node:fs";
import path from "node:path";

const REPO_ROOT = path.join(process.cwd(), "..");
const ARCHIVO = path.join(REPO_ROOT, "GLOSARIO.md");

export type Termino = {
  /** El nombre principal, el que se muestra como título. */
  canonico: string;
  /** Formas alternativas que también se reconocen en el texto. */
  variantes: string[];
  /** El ancla de la URL: /glosario#softmax */
  slug: string;
  /** La definición, todavía en markdown. */
  cuerpo: string;
};

/** Convertir un término en un ancla de URL, sin acentos ni espacios. */
export function slugificar(texto: string): string {
  return texto
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Leer GLOSARIO.md y partirlo en términos.
 *
 * El formato es un "## " por término. El título admite variantes separadas
 * por " / ", y la primera es el nombre principal.
 */
export function leerGlosario(): { intro: string; terminos: Termino[] } {
  if (!fs.existsSync(ARCHIVO)) return { intro: "", terminos: [] };

  const texto = fs.readFileSync(ARCHIVO, "utf8");
  const partes = texto.split(/^## /m);

  // La primera parte es el título de nivel 1 y el párrafo de apertura.
  const intro = partes[0].replace(/^#\s+.*\n+/, "").trim();

  const terminos = partes.slice(1).map((parte) => {
    const corte = parte.indexOf("\n");
    const titulo = (corte === -1 ? parte : parte.slice(0, corte)).trim();
    const cuerpo = (corte === -1 ? "" : parte.slice(corte + 1)).trim();
    const nombres = titulo.split("/").map((n) => n.trim()).filter(Boolean);
    return {
      canonico: nombres[0],
      variantes: nombres.slice(1),
      slug: slugificar(nombres[0]),
      cuerpo,
    };
  });

  return { intro, terminos };
}

/** Todas las formas que hay que buscar en el texto, de la más larga a la más
 *  corta. El orden importa: "multi-head attention" tiene que ganarle a
 *  "attention", si no queda linkeada la palabra suelta adentro de la frase.
 */
export function formasBuscables(terminos: Termino[]): { forma: string; slug: string }[] {
  return terminos
    .flatMap((t) => [t.canonico, ...t.variantes].map((forma) => ({ forma, slug: t.slug })))
    .sort((a, b) => b.forma.length - a.forma.length);
}
