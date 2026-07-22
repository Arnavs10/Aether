/**
 * AETHER — the fifteen emotion chips. Labels arrive from GET /emotions (the
 * order matters for distributions, §2.4); the page owns the fetch and passes
 * them in. Multi mode blends; single mode replaces.
 */

interface Props {
  labels: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  multi?: boolean;
  disabled?: boolean;
}

export function EmotionChips({
  labels,
  selected,
  onChange,
  multi = true,
  disabled = false,
}: Props) {
  const toggle = (label: string) => {
    if (disabled) return;
    if (!multi) {
      onChange(selected[0] === label ? [] : [label]);
      return;
    }
    onChange(
      selected.includes(label)
        ? selected.filter((l) => l !== label)
        : [...selected, label],
    );
  };

  return (
    <div className="flex flex-wrap gap-2">
      {labels.map((label) => {
        const on = selected.includes(label);
        return (
          <button
            key={label}
            type="button"
            onClick={() => toggle(label)}
            disabled={disabled}
            aria-pressed={on}
            className={`mono-meta border px-3.5 py-2.5 transition-all duration-200 disabled:opacity-40 ${
              on
                ? "border-blue text-paper [box-shadow:0_0_18px_rgba(46,107,255,0.35)]"
                : "hairline text-paper/50 hover:border-paper/30 hover:text-paper"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
