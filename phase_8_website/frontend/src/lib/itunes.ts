/**
 * AETHER — iTunes Search delivery layer (§6).
 * The 1.2M store decides WHAT fits; iTunes resolves HOW to hear it:
 * artwork, a 30s previewUrl, and the exact Apple Music link — per track,
 * by `title + artist`, at render time.
 *
 * Constraints honored:
 *  • Apple rate-limits ≈20 req/min/IP → a sequential queue with a ~3.1s
 *    minimum interval, plus a two-layer cache (memory + localStorage) so
 *    repeat renders cost zero requests.
 *  • CORS on itunes.apple.com is historically inconsistent → plain fetch
 *    first, JSONP (`callback=`) fallback second.
 */

export interface ItunesResolved {
  artworkUrl: string | null; // upgraded to 600×600
  previewUrl: string | null; // 30s AAC preview
  appleUrl: string | null; // exact Apple Music page
  durationMs: number | null;
  collection: string | null;
}

const SEARCH_BASE = "https://itunes.apple.com/search";
const MIN_INTERVAL_MS = 3_100; // ≈19/min — safely under Apple's ~20/min
const LS_KEY = "aether.itunes.v1";
const LS_CAP = 300; // entries kept in localStorage

/* ── caches ─────────────────────────────────────────────── */

const memCache = new Map<string, ItunesResolved | null>();

function loadLs(): Record<string, ItunesResolved | null> {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function saveLs(store: Record<string, ItunesResolved | null>): void {
  try {
    const keys = Object.keys(store);
    if (keys.length > LS_CAP) {
      // naive trim: drop oldest-inserted keys
      for (const k of keys.slice(0, keys.length - LS_CAP)) delete store[k];
    }
    localStorage.setItem(LS_KEY, JSON.stringify(store));
  } catch {
    /* private mode / quota — memory cache still works */
  }
}

function cacheKey(title: string, artist: string): string {
  return `${norm(title)}::${norm(artist)}`;
}

/* ── matching helpers ───────────────────────────────────── */

function norm(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N} ]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

interface ItunesRawResult {
  trackName?: string;
  artistName?: string;
  collectionName?: string;
  artworkUrl100?: string;
  previewUrl?: string;
  trackViewUrl?: string;
  trackTimeMillis?: number;
}

/** Score candidates: exact title+artist beats partials; never invent a match. */
function pickBest(
  results: ItunesRawResult[],
  title: string,
  artist: string,
): ItunesRawResult | null {
  const nt = norm(title);
  const na = norm(artist);
  let best: ItunesRawResult | null = null;
  let bestScore = 0;
  for (const r of results) {
    const rt = norm(r.trackName ?? "");
    const ra = norm(r.artistName ?? "");
    let score = 0;
    if (rt === nt) score += 4;
    else if (rt.includes(nt) || nt.includes(rt)) score += 2;
    if (ra === na) score += 4;
    else if (ra.includes(na) || na.includes(ra)) score += 2;
    if (r.previewUrl) score += 1; // playable beats unplayable on ties
    if (score > bestScore) {
      bestScore = score;
      best = r;
    }
  }
  return bestScore >= 3 ? best : null; // require a real match, not noise
}

function toResolved(r: ItunesRawResult): ItunesResolved {
  return {
    artworkUrl: r.artworkUrl100
      ? r.artworkUrl100.replace("100x100", "600x600")
      : null,
    previewUrl: r.previewUrl ?? null,
    appleUrl: r.trackViewUrl ?? null,
    durationMs: r.trackTimeMillis ?? null,
    collection: r.collectionName ?? null,
  };
}

/* ── transport: fetch, then JSONP fallback ──────────────── */

async function searchViaFetch(url: string): Promise<ItunesRawResult[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`itunes ${res.status}`);
  const data = (await res.json()) as { results?: ItunesRawResult[] };
  return data.results ?? [];
}

function searchViaJsonp(url: string): Promise<ItunesRawResult[]> {
  return new Promise((resolve, reject) => {
    const cb = `__aetherItunes${Date.now()}${Math.floor(Math.random() * 1e4)}`;
    const script = document.createElement("script");
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("itunes jsonp timeout"));
    }, 8_000);

    function cleanup() {
      clearTimeout(timer);
      delete (window as unknown as Record<string, unknown>)[cb];
      script.remove();
    }

    (window as unknown as Record<string, unknown>)[cb] = (data: {
      results?: ItunesRawResult[];
    }) => {
      cleanup();
      resolve(data.results ?? []);
    };
    script.onerror = () => {
      cleanup();
      reject(new Error("itunes jsonp failed"));
    };
    script.src = `${url}&callback=${cb}`;
    document.head.appendChild(script);
  });
}

/* ── the priority queue (Pass 4 §8.3) ───────────────────────
   Burst-then-throttle: the first few requests fire with small concurrency
   so the top of a list lands almost immediately, then everything falls
   back to the sequential ~19/min interval Apple tolerates. Viewport code
   boosts priorities as the user scrolls; the scheduler always serves the
   highest-priority pending task next. */

interface QueueTask {
  key: string;
  title: string;
  artist: string;
  priority: number;
  waiters: Array<(r: ItunesResolved | null) => void>;
}

const pendingTasks = new Map<string, QueueTask>();
let inFlight = 0;
let burstRemaining = 4;
const BURST_CONCURRENCY = 2;
let lastRequestAt = 0;
let wakeTimer: ReturnType<typeof setTimeout> | null = null;
let boostSeq = 0;

function pickNext(): QueueTask | null {
  let best: QueueTask | null = null;
  for (const t of pendingTasks.values()) {
    if (!best || t.priority > best.priority) best = t;
  }
  return best;
}

function schedule(): void {
  if (pendingTasks.size === 0) return;
  const now = Date.now();
  const canBurst = burstRemaining > 0 && inFlight < BURST_CONCURRENCY;
  const sequentialReadyAt = lastRequestAt + MIN_INTERVAL_MS;
  const canSequential = inFlight === 0 && now >= sequentialReadyAt;

  if (canBurst || canSequential) {
    const task = pickNext();
    if (!task) return;
    pendingTasks.delete(task.key);
    if (canBurst) burstRemaining--;
    void launch(task);
    schedule(); // a second burst slot may be open
    return;
  }
  if (inFlight === 0 && !wakeTimer) {
    wakeTimer = setTimeout(() => {
      wakeTimer = null;
      schedule();
    }, Math.max(30, sequentialReadyAt - now));
  }
}

async function launch(task: QueueTask): Promise<void> {
  inFlight++;
  lastRequestAt = Date.now();
  const term = encodeURIComponent(`${task.title} ${task.artist}`);
  const url = `${SEARCH_BASE}?term=${term}&media=music&entity=song&limit=5`;

  let resolved: ItunesResolved | null = null;
  try {
    let results: ItunesRawResult[];
    try {
      results = await searchViaFetch(url);
    } catch {
      results = await searchViaJsonp(url); // CORS fallback
    }
    const best = pickBest(results, task.title, task.artist);
    resolved = best ? toResolved(best) : null;
  } catch {
    resolved = null;
  }

  memCache.set(task.key, resolved);
  const store = loadLs();
  store[task.key] = resolved;
  saveLs(store);

  for (const w of task.waiters) w(resolved);
  inFlight--;
  schedule();
}

/* ── public API ─────────────────────────────────────────── */

/**
 * Resolve one track's delivery data. Cached results (including confirmed
 * misses) return instantly and never hit the network. `priority` orders
 * the queue; boostResolve() raises it later (viewport-driven).
 */
export function resolveTrack(
  title: string,
  artist: string,
  priority = 0,
): Promise<ItunesResolved | null> {
  const key = cacheKey(title, artist);
  if (memCache.has(key)) return Promise.resolve(memCache.get(key) ?? null);

  const ls = loadLs();
  if (key in ls) {
    memCache.set(key, ls[key]);
    return Promise.resolve(ls[key]);
  }

  return new Promise((resolve) => {
    const existing = pendingTasks.get(key);
    if (existing) {
      existing.priority = Math.max(existing.priority, priority);
      existing.waiters.push(resolve);
    } else {
      pendingTasks.set(key, {
        key,
        title,
        artist,
        priority,
        waiters: [resolve],
      });
    }
    schedule();
  });
}

/** Raise a pending track to the front of the queue (on-screen cards). */
export function boostResolve(title: string, artist: string): void {
  const task = pendingTasks.get(cacheKey(title, artist));
  if (task) task.priority = 1_000_000 + ++boostSeq;
}

/**
 * Cache-only lookup (no network): returns whatever the resolver has already
 * learned for this track, or null. Used by the playlist exporter so a
 * download reflects everything resolved so far without new requests.
 */
export function peekResolved(title: string, artist: string): ItunesResolved | null {
  const key = cacheKey(title, artist);
  if (memCache.has(key)) return memCache.get(key) ?? null;
  const ls = loadLs();
  return key in ls ? ls[key] : null;
}

/** Spotify / YouTube search deep-links that land on the song (§6). */
export function deepLinks(
  title: string,
  artist: string,
): { spotify: string; youtube: string } {
  const q = encodeURIComponent(`${title} ${artist}`);
  return {
    spotify: `https://open.spotify.com/search/${q}`,
    youtube: `https://www.youtube.com/results?search_query=${q}`,
  };
}
