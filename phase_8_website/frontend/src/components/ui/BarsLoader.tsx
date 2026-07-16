/**
 * AETHER — small stepped-bar loader: the new preloader's visual language in
 * miniature (§12), reused for engine work, voice warming and chat typing.
 * Pure CSS animation (eq-step), so cost is effectively zero.
 */

const DELAYS = [0, 0.12, 0.24, 0.08, 0.18];

export function BarsLoader({
  tone = "silver",
  className = "",
}: {
  tone?: "silver" | "blue" | "gold";
  className?: string;
}) {
  const color =
    tone === "blue"
      ? "var(--color-blue)"
      : tone === "gold"
        ? "var(--color-gold)"
        : "var(--color-silver)";
  return (
    <span
      className={`inline-flex h-4 items-end gap-[3px] ${className}`}
      aria-hidden="true"
    >
      {DELAYS.map((d, i) => (
        <span
          key={i}
          className="eq-step h-full w-[3px]"
          style={{ background: color, opacity: 0.85, animationDelay: `${d}s` }}
        />
      ))}
    </span>
  );
}
