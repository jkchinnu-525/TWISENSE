"use client";

import React from "react";
import { motion } from "framer-motion";
import { Slidetabs } from "./Navbar";

interface HeaderProps {
  text: string;
}
export default function Header({ text }: HeaderProps) {
  return (
    <div className="relative z-0 overflow-hidden pb-24">
      <div
        className="absolute inset-0 z-0"
        style={{
          height: "100%",
          width: "100%",
          background: `
          radial-gradient(ellipse at top, rgba(127, 29, 29, 0.6) 0%, rgba(16, 16, 16, 0) 70%),
          radial-gradient(ellipse at bottom, rgba(16, 16, 16, 0) 0%, rgba(16, 16, 16, 1) 100%),
          linear-gradient(to bottom,
              rgba(127, 29, 29, 0.4) 0%,
              rgba(127, 29, 29, 0.2) 30%,
              rgba(16, 16, 16, 0.8) 60%,
              rgba(16, 16, 16, 1) 100%
            )
          `,
        }}
      />
      <header className="w-full z-10 relative">
        <div className="flex bg-transparent items-center justify-between py-9 mx-auto relative z-10">
          <section className="lg:max-w-screen-xl px-4 mx-auto flex justify-between items-center h-full w-full">
            <motion.div
              whileHover={{ scale: 1.25 }}
              className="flex items-center cursor-pointer "
            >
              {text.split("").map((l, i) => (
                <motion.span
                  key={i}
                  initial={{ x: -100, opacity: 0, scale: 0.8 }}
                  animate={{ x: 0, opacity: 1, scale: 1 }}
                  transition={{
                    duration: 0.4,
                    delay: i * 0.1,
                    type: "spring",
                    stiffness: 100,
                    damping: 15,
                  }}
                  className="text-xl font-semibold"
                >
                  {l}
                </motion.span>
              ))}
            </motion.div>
            <div>
              <Slidetabs></Slidetabs>
            </div>
            <motion.div
              initial={{ x: 100, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.8 }}
              className="flex items-center"
            >
              <a href="https://github.com/jkchinnu-525/TwiSense.git">
                <div className="relative">
                  {/* Animated border */}
                  <motion.div
                    className="absolute -inset-0.5 rounded-md bg-gradient-to-r from-blue-500 to-purple-600 opacity-75 blur-sm"
                    animate={{
                      background: [
                        "linear-gradient(0deg, rgba(59, 130, 246, 0.5), rgba(147, 51, 234, 0.5))",
                        "linear-gradient(90deg, rgba(59, 130, 246, 0.5), rgba(147, 51, 234, 0.5))",
                        "linear-gradient(180deg, rgba(59, 130, 246, 0.5), rgba(147, 51, 234, 0.5))",
                        "linear-gradient(270deg, rgba(59, 130, 246, 0.5), rgba(147, 51, 234, 0.5))",
                        "linear-gradient(360deg, rgba(59, 130, 246, 0.5), rgba(147, 51, 234, 0.5))",
                      ],
                    }}
                    transition={{
                      duration: 4,
                      repeat: Infinity,
                      repeatType: "loop",
                    }}
                  />
                  <button className="relative flex items-center gap-2 px-4 py-4 rounded-md text-white backdrop-blur-sm text-sm border-2 border-white/20 bg-white/10 hover:bg-white/20 cursor-pointer">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      className="lucide lucide-github"
                    >
                      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"></path>
                      <path d="M9 18c-4.51 2-5-2-7-2"></path>
                    </svg>
                    Star us on Github
                  </button>
                </div>
              </a>
            </motion.div>
          </section>
        </div>
      </header>
    </div>
  );
}
