/** AETHER — 404. Quiet, on-brand, one way home. */

import { Link } from "react-router";
import { ParenLabel } from "../components/ui/ParenLabel";

export default function NotFound() {
  return (
    <section className="flex min-h-[70vh] flex-col justify-center px-6 pt-24 md:px-10">
      <ParenLabel>SIGNAL LOST</ParenLabel>
      <h1 className="display mt-6 text-6xl text-paper md:text-8xl">404</h1>
      <p className="serif-accent mt-4 text-xl text-paper/60">
        no track at this address.
      </p>
      <Link
        to="/"
        className="mono-meta mt-10 w-fit border-b border-paper/40 pb-1 text-paper transition-colors hover:border-blue hover:text-blue"
      >
        back to the start →
      </Link>
    </section>
  );
}
