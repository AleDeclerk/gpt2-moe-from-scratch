import fs from "node:fs";
import path from "node:path";

// El sitio vive en web/, y el curso vive un nivel más arriba. Todos los
// números del sitio salen de esos archivos, así que el sitio no tiene estado propio.
const REPO_ROOT = path.join(process.cwd(), "..");

export type TestResult = { name: string; passed: boolean };

export type ChapterStatus = "planned" | "todo" | "in_progress" | "done";

export type Chapter = {
  dir: string;
  title: string;
  writes: string;
  module: string;
  exists: boolean;
  promoted: boolean;
  tests: TestResult[];
  tests_total: number;
  tests_passed: number;
  status: ChapterStatus;
};

export type Part = {
  title: string;
  blurb: string;
  chapters: Chapter[];
};

export type Progress = {
  generated_at: string;
  course: string;
  repo: string;
  totals: {
    chapters_total: number;
    chapters_done: number;
    chapters_written: number;
    tests_total: number;
    tests_passed: number;
  };
  parts: Part[];
};

export function getProgress(): Progress {
  const file = path.join(REPO_ROOT, "progress.json");
  return JSON.parse(fs.readFileSync(file, "utf8")) as Progress;
}

export function getAllChapters(): Chapter[] {
  return getProgress().parts.flatMap((part) => part.chapters);
}

export function getChapter(dir: string): { chapter: Chapter; part: Part; index: number } | null {
  const progress = getProgress();
  const flat = getAllChapters();
  for (const part of progress.parts) {
    const chapter = part.chapters.find((c) => c.dir === dir);
    if (chapter) {
      return { chapter, part, index: flat.findIndex((c) => c.dir === dir) };
    }
  }
  return null;
}

export function getChapterReadme(dir: string): string | null {
  const file = path.join(REPO_ROOT, "chapters", dir, "README.md");
  return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : null;
}

/** El número de un capítulo, sacado del nombre del directorio: ch04_x da "04". */
export function chapterNumber(dir: string): string {
  return dir.slice(2, 4);
}

export function fileUrl(repo: string, dir: string, name: string): string {
  return `${repo}/blob/main/chapters/${dir}/${name}`;
}
