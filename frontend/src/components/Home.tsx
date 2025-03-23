"use client";
import { motion } from "framer-motion";
import React from "react";

export default function Home() {
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
    </div>
  );
}
