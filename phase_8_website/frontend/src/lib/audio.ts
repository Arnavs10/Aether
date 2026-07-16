/**
 * AETHER — single shared preview player (§13). One <audio> element for the
 * whole app so two 30-second previews can never overlap; cards subscribe to
 * know whether their url is the one playing.
 */

let el: HTMLAudioElement | null = null;
let currentUrl: string | null = null;
const listeners = new Set<(url: string | null) => void>();

function emit(): void {
  for (const cb of listeners) cb(currentUrl);
}

function ensure(): HTMLAudioElement {
  if (!el) {
    el = new Audio();
    el.addEventListener("ended", () => {
      currentUrl = null;
      emit();
    });
    el.addEventListener("error", () => {
      currentUrl = null;
      emit();
    });
  }
  return el;
}

/** Play this url, or pause it if it is already the one playing. */
export function togglePreview(url: string): void {
  const audio = ensure();
  if (currentUrl === url) {
    audio.pause();
    currentUrl = null;
    emit();
    return;
  }
  audio.src = url;
  currentUrl = url;
  emit();
  audio.play().catch(() => {
    currentUrl = null;
    emit();
  });
}

export function stopPreview(): void {
  el?.pause();
  currentUrl = null;
  emit();
}

/** Subscribe to playback changes; called immediately with the current url. */
export function subscribePlayback(cb: (url: string | null) => void): () => void {
  listeners.add(cb);
  cb(currentUrl);
  return () => {
    listeners.delete(cb);
  };
}
