/**
 * AETHER — entry point. Fonts are self-hosted via Fontsource (no CDN):
 * Archivo Variable with the WIDTH axis (the display face runs at 125%
 * stretch), Inter Variable for body, Instrument Serif for the editorial
 * accent, Space Mono for the (PARENTHETICAL LABEL) system.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";

import "@fontsource-variable/archivo/wdth.css";
import "@fontsource-variable/inter";
import "@fontsource/instrument-serif";
import "@fontsource/instrument-serif/400-italic.css";
import "@fontsource/space-mono";
import "@fontsource/space-mono/700.css";
import "./styles/index.css";

import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
