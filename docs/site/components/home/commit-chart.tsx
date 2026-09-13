import type { DayCount } from "@/lib/github";

// One series, one hue: commits per day as thin rounded bars on a shared baseline.
// Each bar carries its own title so hovering reads the exact day and count.
export function CommitChart({ days }: { days: DayCount[] }) {
  if (days.length === 0) return null;
  const height = 120;
  const bar = days.length > 20 ? 12 : 20;
  const gap = days.length > 20 ? 4 : 8;
  const width = days.length * bar + (days.length - 1) * gap;
  const max = Math.max(1, ...days.map((d) => d.count));
  const usable = height - 6;
  const label = (date: string) => new Date(date).toLocaleDateString("en", { month: "short", day: "numeric", timeZone: "UTC" });
  return (
    <svg className="commit-chart" viewBox={`0 0 ${width} ${height + 22}`} style={{ maxWidth: width * 1.1 }} role="img" aria-label={`Commits per day over the last ${days.length} days`}>
      <text className="axis" x={0} y={height + 18}>{label(days[0].date)}</text>
      <text className="axis" x={width} y={height + 18} textAnchor="end">{label(days[days.length - 1].date)}</text>
      {days.map((day, index) => {
        const h = day.count === 0 ? 2 : Math.max(4, (day.count / max) * usable);
        const x = index * (bar + gap);
        return (
          <g key={day.date} className={day.count === 0 ? "empty" : undefined}>
            <rect x={x} y={height - h} width={bar} height={h} rx={Math.min(3, bar / 2)} />
            <rect className="hit" x={x - gap / 2} y={0} width={bar + gap} height={height} fill="transparent">
              <title>{`${label(day.date)}: ${day.count} commit${day.count === 1 ? "" : "s"}`}</title>
            </rect>
          </g>
        );
      })}
    </svg>
  );
}
