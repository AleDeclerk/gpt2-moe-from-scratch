import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import Link from "next/link";
import { formasBuscables, leerGlosario } from "@/lib/glosario";
import { renderMarkdown, stripFirstHeading } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "Antes de empezar",
  description: "Qué vas a construir, y el vocabulario mínimo para entender el resto.",
};

function leerIntro(): string | null {
  const archivo = path.join(process.cwd(), "..", "INTRO.md");
  return fs.existsSync(archivo) ? fs.readFileSync(archivo, "utf8") : null;
}

export default async function IntroPage() {
  const source = leerIntro();
  if (!source) {
    return (
      <main className="mx-auto max-w-[1120px] px-6 py-20">
        <p className="text-faint">Todavía no hay INTRO.md en la raíz del repositorio.</p>
      </main>
    );
  }

  const { terminos } = leerGlosario();
  const html = await renderMarkdown(stripFirstHeading(source), formasBuscables(terminos));

  return (
    <main className="mx-auto max-w-[1120px] px-6 pb-24">
      <header className="border-b border-rule py-14">
        <p className="text-[11px] tracking-[0.16em] text-faint uppercase">
          <Link href="/" className="transition-colors hover:text-accent">
            El curso
          </Link>
        </p>
        <h1 className="mt-5 font-display text-[clamp(2.4rem,6vw,3.8rem)] leading-none">
          Antes de empezar
        </h1>
      </header>

      <article
        className="prose mx-auto max-w-[68ch] py-14"
        dangerouslySetInnerHTML={{ __html: html }}
      />

      <nav className="flex justify-end border-t border-rule pt-8 text-[12px]">
        <Link
          href="/chapters/ch00_tensors"
          className="text-faint transition-colors hover:text-accent"
        >
          Empezar por el capítulo 00 &#8594;
        </Link>
      </nav>
    </main>
  );
}
