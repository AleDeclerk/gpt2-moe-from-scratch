import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { StatusChip } from "@/components/Status";
import {
  chapterNumber,
  fileUrl,
  getAllChapters,
  getChapter,
  getChapterReadme,
  getProgress,
} from "@/lib/course";
import { renderMarkdown, stripFirstHeading } from "@/lib/markdown";

type Props = { params: Promise<{ dir: string }> };

export function generateStaticParams() {
  return getAllChapters()
    .filter((chapter) => chapter.exists)
    .map((chapter) => ({ dir: chapter.dir }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { dir } = await params;
  const found = getChapter(dir);
  if (!found) return {};
  return {
    title: `${chapterNumber(dir)} · ${found.chapter.title}`,
    description: found.chapter.writes,
  };
}

export default async function ChapterPage({ params }: Props) {
  const { dir } = await params;
  const found = getChapter(dir);
  const source = getChapterReadme(dir);
  if (!found || !source) notFound();

  const { chapter, part, index } = found;
  const { repo } = getProgress();
  const chapters = getAllChapters();
  const previous = index > 0 ? chapters[index - 1] : null;
  const next = index < chapters.length - 1 ? chapters[index + 1] : null;
  const html = await renderMarkdown(stripFirstHeading(source));

  return (
    <main className="mx-auto max-w-[1120px] px-6 pb-24">
      <header className="border-b border-rule py-14">
        <p className="text-[11px] tracking-[0.16em] text-faint uppercase">
          <Link href="/" className="transition-colors hover:text-accent">
            {part.title}
          </Link>
        </p>
        <div className="mt-5 flex flex-wrap items-end justify-between gap-6">
          <h1 className="flex items-baseline gap-5 font-display text-[clamp(2.1rem,5vw,3.4rem)] leading-none">
            <span className="text-rule tabular-nums">{chapterNumber(dir)}</span>
            <span>{chapter.title}</span>
          </h1>
          <StatusChip status={chapter.status} />
        </div>
        <p className="mt-4 max-w-[60ch] font-prose text-[16px] text-faint">
          Vos escribís: {chapter.writes.toLowerCase()}.
        </p>
      </header>

      <div className="grid gap-14 py-14 lg:grid-cols-[minmax(0,1fr)_17rem] lg:gap-16">
        <article className="prose min-w-0" dangerouslySetInnerHTML={{ __html: html }} />

        <aside className="lg:sticky lg:top-20 lg:h-fit">
          <section className="border border-rule p-5">
            <p className="text-[10px] tracking-[0.18em] text-faint uppercase">
              Tests &#183; {chapter.tests_passed}/{chapter.tests_total}
            </p>
            <ul className="mt-4 space-y-[7px]">
              {chapter.tests.map((test) => (
                <li
                  key={test.name}
                  title={test.name}
                  className="flex gap-2.5 text-[11.5px] leading-snug"
                >
                  <span className={test.passed ? "text-done" : "text-rule"}>
                    {test.passed ? "✓" : "·"}
                  </span>
                  <span className={`[overflow-wrap:anywhere] ${test.passed ? "text-dim" : "text-faint"}`}>
                    {/* Todos los nombres empiezan con test_, así que el prefijo no
                        aporta nada. El atributo title guarda el nombre exacto. */}
                    {test.name.replace(/^test_/, "")}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          <section className="mt-5 border border-rule p-5">
            <p className="text-[10px] tracking-[0.18em] text-faint uppercase">Correlo</p>
            <pre className="mt-3 overflow-x-auto text-[11px] leading-relaxed text-dim">
              <code>{`uv run pytest \\\n  chapters/${dir}`}</code>
            </pre>
            <p className="mt-4 text-[10px] tracking-[0.18em] text-faint uppercase">
              Después promové
            </p>
            <pre className="mt-3 overflow-x-auto text-[11px] leading-relaxed text-dim">
              <code>{`uv run python \\\n  scripts/promote.py ch${chapterNumber(dir)}`}</code>
            </pre>
            <p className="mt-3 text-[11px] text-faint">
              {chapter.promoted
                ? `Promovido. Existe gpt2moe/${chapter.module}.py.`
                : `Todavía sin promover.`}
            </p>
          </section>

          <section className="mt-5 border border-rule p-5">
            <p className="text-[10px] tracking-[0.18em] text-faint uppercase">Archivos</p>
            <ul className="mt-3 space-y-2 text-[12px]">
              {["exercise.py", "test_exercise.py", "solution.py"].map((name) => (
                <li key={name}>
                  <a
                    href={fileUrl(repo, dir, name)}
                    className="text-dim transition-colors hover:text-accent"
                  >
                    {name} &#8599;
                  </a>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </div>

      <nav className="flex flex-wrap justify-between gap-6 border-t border-rule pt-8 text-[12px]">
        {previous?.exists ? (
          <Link
            href={`/chapters/${previous.dir}`}
            className="text-faint transition-colors hover:text-accent"
          >
            &#8592; {chapterNumber(previous.dir)} &#183; {previous.title}
          </Link>
        ) : (
          <span />
        )}
        {next?.exists ? (
          <Link
            href={`/chapters/${next.dir}`}
            className="text-faint transition-colors hover:text-accent"
          >
            {chapterNumber(next.dir)} &#183; {next.title} &#8594;
          </Link>
        ) : (
          next && (
            <span className="text-rule">
              {chapterNumber(next.dir)} &#183; {next.title} &#183; todavía sin escribir
            </span>
          )
        )}
      </nav>
    </main>
  );
}
