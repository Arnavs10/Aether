import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// AETHER frontend build config.
// API base URL comes from VITE_API_BASE (see .env.example) — never hardcoded in components.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
});
