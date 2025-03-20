import Link from "next/link";

export default function Footer() {
  return (
    <div>
      <footer className="relative pt-16 pb-48">
        <div
          className="bottom-0 inset-x-0 z-0"
          style={{
            height: "100%",
            width: "100%",
            pointerEvents: "none",
            position: "absolute",
            background: `
          radial-gradient(ellipse at top, rgba(127, 29, 29, 0.3) 20%, rgba(16, 16, 16, 0) 40%),
          radial-gradient(ellipse at bottom, rgba(16, 16, 16, 0) 80%, rgba(16, 16, 16, 1) 90%),
          linear-gradient(to bottom,
              rgba(127, 29, 29, 0) 0%,
              rgba(127, 29, 29, 0) 10%,
              rgba(16, 16, 16, 0.8) 80%,
              rgba(16, 16, 16, 1) 100%
            )
          `,
          }}
        />
        <div className="flex flex-col justify-center items-center">
          <div>
            <h1 className="text-5xl text-center text-white">
              Sentiment is the <br /> New
              <span className="ml-4 italic">Metric</span>
            </h1>
            <p className="text-xl font-bold py-6">
              AI Doesn’t Just Read Text—It Understands Emotion.
            </p>
          </div>
          <div>
            <div className="flex items-center mt-4">
              <div className="relative inline-flex overflow-hidden p-[2px] focus:outline-none rounded-full">
                <span className="absolute inset-0 animate-[spin_3s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,#E2CBFF_0%,#393BB2_50%,#E2CBFF_100%)]" />
                <div className="flex relative rounded-full bg-neutral-900 justify-center items-center px-8 py-2 gap-4 flex-wrap md:flex-row">
                  <div>
                    <div className="flex items-center gap-2">
                      <svg
                        data-slot="icon"
                        fill="none"
                        strokeWidth="1.9"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        width="16"
                        height="16"
                        xmlns="http://www.w3.org/2000/svg"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941"
                        ></path>
                      </svg>
                      <span className="text-white ">Accurate Analytics</span>
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <svg
                        data-slot="icon"
                        fill="none"
                        strokeWidth="1.9"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        width="16"
                        height="16"
                        xmlns="http://www.w3.org/2000/svg"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941"
                        ></path>
                      </svg>
                      <span className="text-white">Trending Topics</span>
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <svg
                        data-slot="icon"
                        fill="none"
                        strokeWidth="1.5"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        width="16"
                        height="16"
                        xmlns="http://www.w3.org/2000/svg"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"
                        ></path>
                      </svg>
                      <span className="text-white">Visualization</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="flex items-center mt-4">
            <Link href="/features">
              <button className="relative inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm  focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 text-white font-bold h-10 cursor-pointer rounded-md px-8 bg-[#5b2121] hover:from-[#ff3333] hover:to-[#ff5555] mt-6 transform hover:scale-105 transition-all duration-200">
                Get Started
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="lucide lucide-arrow-right ml-2 size-4"
                >
                  <path d="M5 12h14"></path>
                  <path d="m12 5 7 7-7 7"></path>
                </svg>
              </button>
            </Link>
          </div>
        </div>
      </footer>
      <p className="text-sm mt-8 md:mt-0 text-muted-foreground pb-2 text-center">
        © Twisense. All rights reserved.
      </p>
    </div>
  );
}
