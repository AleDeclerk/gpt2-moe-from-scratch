import type { Metadata } from "next";
import Link from "next/link";
import { leerGlosario } from "@/lib/glosario";
import { renderMarkdown } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "Glosario",
  description: "Cada término técnico del curso, definido en dos o tres frases.",
};

export default async function GlosarioPage() {
  const { intro, terminos } = leerGlosario();

  // Las definiciones no enlazan a otros términos: la página ya es el glosario,
  // y enlazarse a sí misma sería ruido.
  const introHtml = intro ? await renderMarkdown(intro) : "";
  const definiciones = await Promise.all(
    terminos.map(async (t) => ({ ...t, html: await renderMarkdown(t.cuerpo) })),
  );

  const iniciales = [...new Set(terminos.map((t) => t.slug[0]?.toUpperCase()).filter(Boolean))];

  return (
    <main className="mx-auto max-w-[1120px] px-6 pb-24">
      <header className="border-b border-rule py-14">
        <p className="text-[11px] tracking-[0.16em] text-faint uppercase">
          <Link href="/" className="transition-colors hover:text-accent">
            El curso
          </Link>
        </p>
        <h1 className="mt-5 font-display text-[clamp(2.4rem,6vw,3.8rem)] leading-none">
          Glosario
        </h1>
        {introHtml ? (
          <div
            className="prose mt-6 max-w-[60ch]"
            dangerouslySetInnerHTML={{ __html: introHtml }}
          />
        ) : null}
      </header>

      {iniciales.length > 0 ? (
        <nav className="flex flex-wrap gap-x-3 gap-y-2 border-b border-rule py-6 text-[12px]">
          {iniciales.map((letra) => (
            <a
              key={letra}
              href={`#letra-${letra}`}
              className="text-faint transition-colors hover:text-accent"
            >
              {letra}
            </a>
          ))}
        </nav>
      ) : null}

      <dl className="py-6">
        {definiciones.map((t, i) => {
          const letra = t.slug[0]?.toUpperCase();
          const primeraDeSuLetra = definiciones[i - 1]?.slug[0]?.toUpperCase() !== letra;
          return (
            <div
              key={t.slug}
              id={t.slug}
              className="scroll-mt-20 border-t border-rule/70 py-6 md:grid md:grid-cols-[14rem_minmax(0,1fr)] md:gap-8"
            >
              <dt id={primeraDeSuLetra ? `letra-${letra}` : undefined} className="scroll-mt-20">
                <span className="text-[15px] text-ink">{t.canonico}</span>
                {t.variantes.length > 0 ? (
                  <span className="mt-1 block text-[11px] text-faint">
                    también: {t.variantes.join(", ")}
                  </span>
                ) : null}
              </dt>
              <dd
                className="prose mt-2 max-w-[62ch] text-[16px] md:mt-0"
                dangerouslySetInnerHTML={{ __html: t.html }}
              />
            </div>
          );
        })}
      </dl>

      {definiciones.length === 0 ? (
        <p className="py-10 text-faint">Todavía no hay GLOSARIO.md en la raíz del repositorio.</p>
      ) : null}
    </main>
  );
}
