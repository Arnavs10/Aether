/**
 * AETHER — FAQ accordion (§5.3 rewrite). Same coverage, plainer language:
 * what each page does, login is never required, what login will add, how
 * voice works, what happens to data, how songs are chosen. Conservative
 * data-privacy wording kept. Accordion mechanics unchanged.
 */

import { useId, useState } from "react";

interface FaqItem {
  q: string;
  a: string;
}

const ITEMS: FaqItem[] = [
  {
    q: "What do the four features actually do?",
    a: "Curate is the core. Tell it how you feel, typed or spoken, in English or Hindi, and it builds a playlist from 1.2 million songs with the reasoning shown for every pick. Journey plans a route: say where you are and where you want to end up, and it arranges the stops between. Live listens while music plays and, when your mood moves, mixes into a song that fits the new one. Connect is Arnav, and a way to reach him.",
  },
  {
    q: "Do I need an account or login?",
    a: "No. Everything here works without one. Curation, journeys, live mixing, voice input, downloads. There is nothing to sign up for.",
  },
  {
    q: "What will logging in add, later?",
    a: "Signing in with a streaming account will eventually add full songs playing right on the page, plus one-tap saves to your own library. Until then you get 30 second previews and direct links that open each song on Apple Music, Spotify or YouTube.",
  },
  {
    q: "How does voice input work?",
    a: "Tap the mic and talk normally, in English or Hindi. One model listens to how you sound while another writes down what you said, and together they land on the feeling. They warm up quietly in the background when you open a page, so the mic is usually ready by the time you are.",
  },
  {
    q: "What happens to my data?",
    a: "There are no accounts and no profile. What you type or say goes to Aether only to work out the feeling for that one request. If you curate something, the single emotion word may appear anonymously in the public feed on the Connect page. Messages from the contact form go straight to Arnav's inbox.",
  },
  {
    q: "How does it actually choose songs?",
    a: "Your feeling becomes a target. Every song in the library has a measurable character: its pace, its energy, how bright or dark it sits. The songs closest to your target surface first, get arranged so the energy moves with intention, and each one can explain why it made the cut.",
  },
];

export function FaqAccordion() {
  const [openIdx, setOpenIdx] = useState<number | null>(0);
  const baseId = useId();

  return (
    <div className="divide-y divide-paper/8 border-y hairline">
      {ITEMS.map((item, i) => {
        const open = openIdx === i;
        const panelId = `${baseId}-panel-${i}`;
        const buttonId = `${baseId}-button-${i}`;
        return (
          <div key={item.q}>
            <button
              type="button"
              id={buttonId}
              aria-expanded={open}
              aria-controls={panelId}
              onClick={() => setOpenIdx(open ? null : i)}
              className="flex w-full items-baseline justify-between gap-6 py-5 text-left"
            >
              <span className="text-base font-medium text-paper md:text-lg">
                {item.q}
              </span>
              <span
                aria-hidden="true"
                className={`mono-meta shrink-0 text-paper/40 transition-transform duration-500 ${
                  open ? "rotate-45" : ""
                }`}
              >
                +
              </span>
            </button>
            <div
              id={panelId}
              role="region"
              aria-labelledby={buttonId}
              className="grid transition-[grid-template-rows] duration-500 ease-out"
              style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
            >
              <div className="overflow-hidden">
                <p className="max-w-2xl pb-6 text-sm leading-relaxed text-mist">
                  {item.a}
                </p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
