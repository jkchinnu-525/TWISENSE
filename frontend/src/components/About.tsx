"use client";
import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface Text {
  direction: string;
}

export default function About({ direction }: Text) {
  const ref = useRef(null);
  const InView = useInView(ref, { once: false });
  const Text =
    "In a world where opinions spread faster than ever, Social Sentiment Dashboard transforms the chaos of social media into actionable insights. We analyze millions of tweets and Reddit posts in real time to measure public sentiment on trending topics, From Cryptocurrencies To Blockbuster Movies.";
  const Text_Fade = {
    show: { opacity: 1, y: 0, transition: { type: "spring" } },
    hide: { opacity: 0, y: direction === "down" ? -18 : 18 },
  };
  return (
    <div className="relative z-10 pb-64">
      <div className="flex items-center justify-center py-4">
        <motion.div
          ref={ref}
          initial="hide"
          animate={InView ? "show" : ""}
          variants={Text_Fade}
          className="text-center"
        >
          <h1 className="text-5xl font-bold">
            See Beyond the Hashtags. Understand
          </h1>
          <h1 className="text-5xl font-bold"> What the World Really Thinks.</h1>
        </motion.div>
      </div>
      <motion.div
        ref={ref}
        initial={{ x: -100, opacity: 0 }}
        animate={InView ? { x: 0, opacity: 1 } : ""}
        transition={{ type: "spring", duration: 2 }}
        className="text-xl font-semibold px-28 py-16"
      >
        <p className="flex items-center justify-center bg-clip-text text-transparent bg-gradient-to-r from-white to-neutral-500">
          {Text}
        </p>
      </motion.div>
    </div>
  );
}
