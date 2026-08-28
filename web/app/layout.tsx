import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Serif, Source_Serif_4 } from "next/font/google";
import Link from "next/link";
import { getProgress } from "@/lib/course";
import "./globals.css";

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const instrument = Instrument_Serif({
  variable: "--font-instrument",
  subsets: ["latin"],
  weight: "400",
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GPT-2 with Mixture of Experts, from scratch",
  description:
    "A course in the form of a repository. You write the code, the tests give the pass condition, and this site reads the result.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const { totals, repo } = getProgress();

  return (
    // The font variables must sit on <html>. The theme in globals.css defines
    // --font-display in :root, and it reads these. A definition on <body>
    // would be invisible from :root, and every font would fall back.
    <html lang="en" className={`${plexMono.variable} ${instrument.variable} ${sourceSerif.variable}`}>
      <body>
        <header className="sticky top-0 z-50 border-b border-rule bg-paper/85 backdrop-blur-sm">
          <div className="mx-auto flex max-w-[1120px] items-center justify-between gap-4 px-6 py-3 text-[11px] tracking-[0.14em] uppercase">
            <Link href="/" className="text-ink transition-colors hover:text-accent">
              gpt2-moe
            </Link>
            <div className="flex items-center gap-5 text-faint">
              <span>
                <span className="text-dim tabular-nums">
                  {totals.tests_passed}/{totals.tests_total}
                </span>{" "}
                tests
              </span>
              <a href={repo} className="transition-colors hover:text-accent">
                repo &#8599;
              </a>
            </div>
          </div>
        </header>
        {children}
        <footer className="mx-auto max-w-[1120px] px-6 py-14 text-[11px] tracking-[0.12em] text-faint uppercase">
          Every number on this site comes from a pytest run in the repository.
        </footer>
      </body>
    </html>
  );
}
