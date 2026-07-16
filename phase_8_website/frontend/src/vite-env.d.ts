/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API base URL for the Aether FastAPI service. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
