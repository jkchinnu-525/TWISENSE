"use client";

import Template from "@/components/ui/transition";
import { motion } from "framer-motion";
import { PlaceholdersAndVanishInput } from "@/components/ui/InputBox";
export default function Feature1() {
  const placeholders = [
    "The food at that restaurant was amazing!",

    "I’m so excited for the weekend!",

    "This is the worst day ever.",

    "I’m feeling really optimistic about the future.",

    "Why is traffic so bad today?",
  ];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);
  };
  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log("submitted");
  };

  const text = "Analyze Sentiment In Real Time";
  const text_effect = {
    initial: { y: 20, opacity: 0 },
    animate: (i: number) => ({
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.8,
        ease: "easeInOut",
        delay: i * 0.1,
      },
    }),
  };
  return (
    <Template>
      <div className="flex justify-center">
        {text.split(" ").map((cur, i) => (
          <motion.div
            key={i}
            variants={text_effect}
            initial="initial"
            custom={i}
            animate="animate"
            className="text-5xl md:leading-[4rem] font-medium text-center tracking-tighter pr-2 mt-24  bg-clip-text bg-gradient-to-b from-[#ffffff] to-neutral-400 text-transparent"
          >
            {cur == "" ? <span>&nbsp;</span> : cur}
          </motion.div>
        ))}
      </div>
      <div className="mt-20">
        <motion.div
          initial={{ x: -100, opacity: 0, scale: 0.8 }}
          animate={{ x: 0, opacity: 1, scale: 1 }}
          transition={{ duration: 1 }}
          whileHover={{ scale: 1.05 }}
          className="text-xl mb-20 cursor-pointer font-semibold flex flex-col justify-center items-center text-center text-gray-400"
        >
          Enter any text, and let our AI-powered model
          <br />
          decode the emotions behind it.
        </motion.div>
        <div>
          <PlaceholdersAndVanishInput
            placeholders={placeholders}
            onChange={handleChange}
            onSubmit={onSubmit}
          />
        </div>
      </div>
    </Template>
  );
}
