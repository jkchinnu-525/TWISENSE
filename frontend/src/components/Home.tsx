"use client";
import { motion, useInView } from "framer-motion";
import React, { useRef } from "react";

export default function Home() {
  // const ref = useRef(null);
  // const Inview = useInView(ref, { once: true, amount: "some" });
  // const dir = direction || "down";
  // const text_effect = {
  //   show: {
  //     opacity: 1,
  //     y: 0,
  //     scale: 1,
  //     transition: { type: "spring", delay: 0.2, duration: 1.4 },
  //   },
  //   hide: { opacity: 0, y: dir === "down" ? -60 : 60, scale: 0.8 },
  // };
  return (
    <div>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.8 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 1.2, delay: 0.1 }}
        whileHover={{ scale: 1.1 }}
        className="text-center my-24 pb-32 lg:pb-44 2xl:pb-80"
      >
        <span className="cursor-pointer">
          <h1 className="text-5xl  cursor-pointer md:text-6xl font-semibold">
            Decoding Emotions By,
          </h1>
          <h1 className="text-5xl md:text-6xl font-semibold">
            One Tweet at a Time
          </h1>
          <p className="text-xl mt-6 font-medium">
            Discover the power of understanding emotions through the lens of a
            single tweet.
          </p>
          <p className="text-lg font-medium">
            The Pulse of Public Opinion, Powered by AI.
          </p>
        </span>
      </motion.div>
      {/* <motion.div
        ref={ref}
        initial="hide"
        animate={Inview ? "show" : "hide"}
        variants={text_effect}
        className="pt-44 flex items-center flex-col text-center justify-center"
      >
        <h2 className="bg-clip-text bg-gradient-to-b from-[#ffffff] to-neutral-400 font-medium text-transparent text-2xl md:text-4xl lg:text-5xl !leading-tight">
          Unlock Data-Driven Insights
          <br />
          with Cutting-Edge
          <span className="ml-2 font-subheading italic">Analytic</span>
        </h2>
        <p className="text-center mx-auto max-w-xl pt-6 font-extralight">
          Track user behavior, monitor key metrics, and optimize performance
          effortlessly with real-time analytics.
        </p>
      </motion.div> */}
    </div>
  );
}
