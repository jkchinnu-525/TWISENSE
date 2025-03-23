"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import Template from "@/components/ui/transition";
import { motion } from "framer-motion";
import { Skeleton } from "@/components/ui/skeleton";
import { PlaceholdersAndVanishInput } from "@/components/ui/InputBox";
import Link from "next/link";

export default function Feature1() {
  const [sentimentResult, setSentimentResult] = useState<{
    sentiment: string;
    confidence: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [inputText, setInputText] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
  };

  const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_URL,
    headers: {
      "Content-Type": "application/json",
    },
  });

  const analyzeSentiment = async (text: string) => {
    setLoading(true);
    try {
      const res = await api.post("/predict", { text });
      setSentimentResult(res.data);
      return res.data;
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const error_msg =
          err.response?.data?.detail || "Server is Busy, Try again later";
        setErr(error_msg);
        console.log("API Error: ", error_msg);
      } else {
        setErr("An unexpected error occurred");
        console.log("Unexpected Error: ", err);
      }
      return null;
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    await analyzeSentiment(inputText);
  };

  const placeholders = [
    "The food at that restaurant was amazing!",

    "I’m so excited for the weekend!",

    "This is the worst day ever.",

    "I’m feeling really optimistic about the future.",

    "Why is traffic so bad today?",
  ];

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

  useEffect(() => {
    setSentimentResult(null);
    setLoading(false);
    setInputText("");
    setErr(null);
  }, []);
  return (
    <Template>
      <motion.div
        initial={{ x: -100, opacity: 0, scale: 0.8 }}
        animate={{ x: 0, opacity: 1, scale: 1 }}
        transition={{ duration: 1 }}
        whileHover={{ scale: 1.05 }}
        className="ml-8 pt-6 "
      >
        <Link href="/features">
          <button className="inline-flex items-center justify-center cursor-pointer gap-2 text-neutral-400 italic  text-2xl">
            <svg
              data-slot="icon"
              fill="none"
              width="24"
              height="24"
              strokeWidth="1.5"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18"
              ></path>
            </svg>
            Go Back
          </button>
        </Link>
      </motion.div>
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
            onSubmit={handleSubmit}
          />
        </div>
        {err && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="flex justify-center mt-6"
          >
            <div className="bg-red-900/20 border border-red-500/30 text-red-200 px-4 py-3 rounded-full max-w-md w-full">
              <p className="text-center">{err}</p>
            </div>
          </motion.div>
        )}

        {loading ? (
          <div className="flex justify-center mt-8">
            <Skeleton className="w-[450px] h-[120px] rounded-full" />
          </div>
        ) : (
          sentimentResult && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="flex items-center justify-center mt-12"
            >
              <div className="relative inline-flex overflow-hidden p-[2px]  focus:outline-none rounded-full ">
                <span className="absolute inset-[-1000%] animate-[spin_3s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,#E2CBFF_0%,#393BB2_50%,#E2CBFF_100%)]" />
                <div className="flex relative md:flex-row flex-wrap rounded-full justify-center items-center p-6 px-28 bg-black ">
                  <h3 className="text-2xl font-bold mb-2 text-center">
                    {sentimentResult.sentiment === "POSITIVE"
                      ? "😊 Positive"
                      : "😔 Negative"}
                  </h3>
                  <div className="w-full bg-gray-700/30 rounded-full h-2.5 mb-2">
                    <div
                      className={`h-2.5 rounded-full ${
                        sentimentResult.sentiment === "POSITIVE"
                          ? "bg-green-500"
                          : "bg-red-500"
                      }`}
                      style={{ width: `${sentimentResult.confidence * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-center text-gray-400">
                    Confidence: {(sentimentResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </motion.div>
          )
        )}
      </div>
    </Template>
  );
}
