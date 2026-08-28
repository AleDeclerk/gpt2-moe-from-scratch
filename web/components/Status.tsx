import type { Chapter, ChapterStatus } from "@/lib/course";

const LABEL: Record<ChapterStatus, string> = {
  planned: "planned",
  todo: "not started",
  in_progress: "in progress",
  done: "done",
};

const TONE: Record<ChapterStatus, string> = {
  planned: "border-rule text-faint border-dashed",
  todo: "border-rule text-dim",
  in_progress: "border-progress/50 text-progress",
  done: "border-done/50 text-done",
};

export function StatusChip({ status }: { status: ChapterStatus }) {
  return (
    <span
      className={`inline-block shrink-0 rounded-full border px-2.5 py-[3px] text-[10px] tracking-[0.12em] uppercase ${TONE[status]}`}
    >
      {LABEL[status]}
    </span>
  );
}

/** One mark for each test of the chapter. The bar is the chapter, in detail. */
export function TestBar({ chapter }: { chapter: Chapter }) {
  if (!chapter.exists) {
    return <span className="text-[11px] text-faint">not written yet</span>;
  }
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex gap-[3px]">
        {chapter.tests.map((test) => (
          <span
            key={test.name}
            title={`${test.name} — ${test.passed ? "passes" : "fails"}`}
            className={`h-3 w-[5px] rounded-[1px] ${test.passed ? "bg-done" : "bg-rule"}`}
          />
        ))}
      </div>
      <span className="text-[11px] tabular-nums text-faint">
        {chapter.tests_passed}/{chapter.tests_total}
      </span>
    </div>
  );
}
