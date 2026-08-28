import Link from "next/link";
import { StatusChip, TestBar } from "@/components/Status";
import { chapterNumber, getProgress } from "@/lib/course";

export default function Home() {
  const { totals, parts, repo, generated_at } = getProgress();
  const percent = totals.tests_total
    ? Math.round((totals.tests_passed / totals.tests_total) * 100)
    : 0;

  let row = 0;

  return (
    <main className="mx-auto max-w-[1120px] px-6">
      {/* ---- hero ---- */}
      <section className="grid gap-12 border-b border-rule py-20 md:grid-cols-[1.45fr_1fr] md:gap-16 md:py-28">
        <div>
          <p className="rise mb-6 text-[11px] tracking-[0.2em] text-accent uppercase">
            A course you run, not a course you watch
          </p>
          <h1
            className="rise font-[family-name:var(--font-display)] text-[clamp(2.7rem,7vw,4.6rem)] leading-[0.95] tracking-[-0.01em]"
            style={{ "--i": 1 } as React.CSSProperties}
          >
            GPT&#8209;2 with a<br />
            Mixture of&nbsp;Experts,
            <br />
            <span className="text-faint italic">from scratch.</span>
          </h1>
          <p
            className="rise mt-8 max-w-[52ch] font-[family-name:var(--font-prose)] text-[17px] leading-[1.7] text-dim"
            style={{ "--i": 2 } as React.CSSProperties}
          >
            Fourteen chapters that build a language model twice. First the dense
            GPT&#8209;2 of 2019. Then the sparse variant behind Mixtral and
            DeepSeek. Every function is written by hand, and pytest gives the
            pass condition.
          </p>
        </div>

        {/* ---- the instrument ---- */}
        <aside
          className="rise self-end border border-rule bg-raised/60 p-6"
          style={{ "--i": 3 } as React.CSSProperties}
        >
          <p className="text-[10px] tracking-[0.18em] text-faint uppercase">
            Progress, measured
          </p>
          <p className="mt-5 font-[family-name:var(--font-display)] text-6xl leading-none tabular-nums">
            {totals.tests_passed}
            <span className="text-2xl text-faint">/{totals.tests_total}</span>
          </p>
          <p className="mt-2 text-[11px] tracking-[0.12em] text-faint uppercase">
            tests green
          </p>

          <div className="mt-6 h-[3px] w-full bg-rule">
            <div
              className="h-full bg-accent transition-[width] duration-700"
              style={{ width: `${percent}%` }}
            />
          </div>

          <dl className="mt-6 space-y-2 text-[11px] tracking-[0.1em] uppercase">
            <div className="flex justify-between">
              <dt className="text-faint">chapters done</dt>
              <dd className="tabular-nums text-dim">
                {totals.chapters_done}/{totals.chapters_total}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-faint">chapters written</dt>
              <dd className="tabular-nums text-dim">
                {totals.chapters_written}/{totals.chapters_total}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-faint">measured</dt>
              <dd className="text-dim">{generated_at.slice(0, 10)}</dd>
            </div>
          </dl>
        </aside>
      </section>

      {/* ---- how it works ---- */}
      <section className="grid gap-8 border-b border-rule py-14 sm:grid-cols-3">
        {[
          {
            n: "01",
            head: "Read, then write",
            body: "Each chapter explains the theory, then hands you a file of empty functions.",
          },
          {
            n: "02",
            head: "The tests decide",
            body: "Green is not an opinion. Run pytest, and the chapter tells you what is still wrong.",
          },
          {
            n: "03",
            head: "Promote your code",
            body: "Validated code moves into the package, and the next chapter imports it. You build the library.",
          },
        ].map((step, i) => (
          <div key={step.n} className="rise" style={{ "--i": i } as React.CSSProperties}>
            <p className="text-[11px] tracking-[0.18em] text-accent">{step.n}</p>
            <h2 className="mt-3 font-[family-name:var(--font-display)] text-xl">{step.head}</h2>
            <p className="mt-2 max-w-[38ch] font-[family-name:var(--font-prose)] text-[15px] leading-relaxed text-faint">
              {step.body}
            </p>
          </div>
        ))}
      </section>

      {/* ---- the chapters ---- */}
      {parts.map((part, partIndex) => (
        <section key={part.title} className="border-b border-rule py-14">
          <div className="mb-8 flex flex-col gap-2 sm:flex-row sm:items-baseline sm:gap-6">
            <span className="font-[family-name:var(--font-display)] text-4xl text-rule tabular-nums">
              {String(partIndex + 1).padStart(2, "0")}
            </span>
            <div>
              <h2 className="font-[family-name:var(--font-display)] text-2xl">{part.title}</h2>
              <p className="mt-1 max-w-[62ch] font-[family-name:var(--font-prose)] text-[15px] text-faint">
                {part.blurb}
              </p>
            </div>
          </div>

          <ul>
            {part.chapters.map((chapter) => {
              const inner = (
                <div className="grid grid-cols-[2.6rem_1fr] items-baseline gap-x-4 gap-y-3 py-4 md:grid-cols-[2.6rem_minmax(0,1fr)_auto_7.5rem] md:items-center">
                  <span className="font-[family-name:var(--font-display)] text-2xl text-faint tabular-nums transition-colors group-hover:text-accent">
                    {chapterNumber(chapter.dir)}
                  </span>
                  <div className="min-w-0">
                    <p className="truncate text-[15px] text-ink">{chapter.title}</p>
                    <p className="mt-1 truncate font-[family-name:var(--font-prose)] text-[14px] text-faint">
                      {chapter.writes}
                    </p>
                  </div>
                  <div className="col-start-2 md:col-start-auto">
                    <TestBar chapter={chapter} />
                  </div>
                  <div className="col-start-2 md:col-start-auto md:justify-self-end">
                    <StatusChip status={chapter.status} />
                  </div>
                </div>
              );

              row += 1;
              const style = { "--i": row } as React.CSSProperties;

              return (
                <li key={chapter.dir} className="rise border-t border-rule/70" style={style}>
                  {chapter.exists ? (
                    <Link
                      href={`/chapters/${chapter.dir}`}
                      className="group block transition-colors hover:bg-raised/60"
                    >
                      {inner}
                    </Link>
                  ) : (
                    <div className="opacity-45">{inner}</div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      ))}

      <section className="py-14">
        <a
          href={repo}
          className="inline-block border border-rule px-5 py-3 text-[11px] tracking-[0.16em] uppercase transition-colors hover:border-accent hover:text-accent"
        >
          Clone the repository &#8599;
        </a>
      </section>
    </main>
  );
}
