/**
 * AETHER — application shell.
 * Layer order (bottom → top): AmbientWaves (z-0) → page content (z-10) →
 * GrainOverlay (z-60) → ScrollProgress (z-70) → Nav (z-80) → mobile menu
 * (z-90) → Preloader (z-100). The preloader gates appReady; the hero
 * entrance keys off it. A degraded engine browses silently (§3.5): any
 * failure shows one calm line inside the panel that failed, never globally.
 */

import { Route, Routes } from "react-router";
import { AppStateProvider } from "./state/AppState";
import { Preloader } from "./components/global/Preloader";
import { SmoothScroll } from "./components/global/SmoothScroll";
import { AmbientWaves } from "./components/global/AmbientWaves";
import { GrainOverlay } from "./components/global/GrainOverlay";
import { ScrollProgress } from "./components/global/ScrollProgress";
import { Nav } from "./components/global/Nav";
import { Footer } from "./components/global/Footer";
import { Chatbot } from "./components/global/Chatbot";
import Home from "./pages/Home";
import Curate from "./pages/Curate";
import Journey from "./pages/Journey";
import Live from "./pages/Live";
import Connect from "./pages/Connect";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <AppStateProvider>
      <Preloader />
      <SmoothScroll />
      <AmbientWaves />
      <ScrollProgress />
      <Nav />

      <main className="relative z-10">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/curate" element={<Curate />} />
          <Route path="/journey" element={<Journey />} />
          <Route path="/live" element={<Live />} />
          <Route path="/connect" element={<Connect />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
        <Footer />
      </main>

      <Chatbot />
      <GrainOverlay />
    </AppStateProvider>
  );
}
