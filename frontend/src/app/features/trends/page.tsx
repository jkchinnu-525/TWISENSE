"use client";
import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { PlaceholdersAndVanishInput } from "@/components/ui/InputBox";
import Template from "@/components/ui/transition";
import Link from "next/link";
import axios from "axios";
import { Skeleton } from "@/components/ui/skeleton";
import ScrollEffect from "@/components/ui/ScrollingEffect";
import jsPDF from "jspdf";

interface TrendResult {
  text: string;
  username: string;
}
export default function Feature2() {
  const [tweets, setTweets] = useState<TrendResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [inputText, setInputText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [lastSearchTerm, setLastSearchTerm] = useState("");
  const [generating, setGenerating] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
    setErr(null);
  };

  const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_URL,
    headers: {
      "Content-Type": "application/json",
    },
  });

  const fetchTweets = async (query: string) => {
    setLoading(true);
    try {
      const res = await api.get(`/tweets?query=${encodeURIComponent(query)}`);
      const extractedTweets = res.data.map((tweet: TrendResult) => ({
        text: tweet.text,
        username: tweet.username,
      }));
      setTweets(extractedTweets);
      return extractedTweets;
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const error_msg =
          err.response?.data?.detail || "Failed to fetch Tweets";
        setErr(error_msg);
      } else {
        setErr("An unexpected error occurred");
      }
      return [];
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    setLastSearchTerm(inputText);
    await fetchTweets(inputText);
  };

  const placeholders = [
    "Bitcoin price surge",

    "Elon Musk latest tweet",

    "World Cup 2023 highlights",

    "AI advancements in 2023",

    "Crypto market trends",
  ];

  const text = "Explore Real-Time Tweets";
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
    setTweets([]);
    setLoading(false);
    setInputText("");
    setErr(null);
    setLastSearchTerm("");
  }, []);

  const generatePDF = () => {
    if (tweets.length === 0) return;

    setGenerating(true);
    try {
      const doc = new jsPDF();

      // Set title
      doc.setFontSize(20);
      doc.setTextColor(33, 33, 33);
      doc.text(`Tweets about "${lastSearchTerm}"`, 20, 20);
      doc.setDrawColor(100, 100, 240);
      doc.line(20, 25, 190, 25);

      // Setting content formatting
      doc.setFontSize(12);
      doc.setTextColor(33, 33, 33);

      let yPos = 40;

      tweets.forEach((tweet, index) => {
        // Adding more pages if content exceeds page height
        if (yPos > 270) {
          doc.addPage();
          yPos = 20;
        }

        // Username
        doc.setFontSize(13);
        doc.setTextColor(0, 102, 204);
        doc.text(`@${tweet.username}`, 20, yPos);
        yPos += 7;

        // Tweet text - handle long text with wrapping
        doc.setFontSize(11);
        doc.setTextColor(33, 33, 33);

        const splitText = doc.splitTextToSize(tweet.text, 170);
        doc.text(splitText, 20, yPos);

        // Calculating position for next tweet based on text height
        yPos += splitText.length * 7 + 15;

        // Adding separator line between tweets
        if (index < tweets.length - 1) {
          doc.setDrawColor(220, 220, 220);
          doc.line(20, yPos - 8, 190, yPos - 8);
        }
      });

      // Adding generation date at the bottom of the last page
      const today = new Date();
      const dateString = today.toLocaleString();
      doc.setFontSize(9);
      doc.setTextColor(150, 150, 150);
      doc.text(`Generated on ${dateString} via TwiSense`, 20, 290);

      // Save the PDF
      doc.save(`tweets-${lastSearchTerm.replace(/\s+/g, "-")}.pdf`);
    } catch (error) {
      console.error("Error generating PDF:", error);
      setErr("Failed to generate PDF. Please try again.");
    } finally {
      setGenerating(false);
    }
  };
  return (
    <ScrollEffect>
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
            See what the world is saying about any
            <br />
            topic—live and unfiltered.
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
              <div className="bg-red-900/20 border border-red-500/30 text-red-200 px-4 py-3 rounded-lg max-w-md w-full">
                <p className="text-center">{err}</p>
              </div>
            </motion.div>
          )}
          {loading ? (
            <div className="flex justify-center mt-8">
              <Skeleton className="w-[450px] h-[120px] rounded-full" />
            </div>
          ) : (
            tweets.length > 0 && (
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="flex flex-col items-center justify-center mt-12 w-full max-w-2xl mx-auto"
              >
                <div className="relative w-full">
                  <div className="absolute inset-[-2px] rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 opacity-75 blur-sm"></div>
                  <div className="relative bg-black rounded-xl overflow-hidden p-4 w-full">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-xl font-semibold text-white mb-4">
                        Latest Tweets about &quot;{lastSearchTerm}&quot;
                      </h3>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={generatePDF}
                        disabled={generating}
                        className="flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-4 py-2 rounded-lg shadow-md transition-all hover:shadow-xl"
                      >
                        {generating ? (
                          <>
                            <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-1"></span>
                            Generating...
                          </>
                        ) : (
                          <div className="flex gap-2 cursor-pointer">
                            <svg
                              width="24"
                              height="24"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M12 16L12 8"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              <path
                                d="M9 13L12 16L15 13"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              <path
                                d="M20 16.7C20 17.4333 19.7366 18.1167 19.2678 18.6167C18.7989 19.1167 18.163 19.4229 17.5 19.45C16.7718 19.4729 16.0849 19.1732 15.5767 18.6175C15.0685 18.0617 14.7781 17.3212 14.775 16.55V16.25C14.7719 15.6035 14.5323 14.9823 14.1045 14.5022C13.6767 14.0222 13.088 13.7164 12.45 13.65C11.96 13.6 11.3 13.6 10.6 13.6H8.6"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              <path
                                d="M8 21H7.4C6.08 21 5.42 21 4.908 20.782C4.32893 20.5305 3.83914 20.111 3.5 19.575C3.2 19.092 3 18.472 3 17.232V9.768C3 8.529 3.2 7.908 3.5 7.425C3.83914 6.88905 4.32893 6.46947 4.908 6.218C5.42 6 6.08 6 7.4 6H16.6C17.92 6 18.58 6 19.092 6.218C19.6711 6.46947 20.1609 6.88905 20.5 7.425C20.8 7.908 21 8.529 21 9.768V12.5"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                            Download PDF
                          </div>
                        )}
                      </motion.button>
                    </div>
                    <div className="space-y-4 overflow-y-auto scrollbar-thin">
                      {tweets.map((tweet, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="bg-gray-900/50 p-4 rounded-lg"
                        >
                          <div className="flex items-center mb-2">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-400 to-indigo-500 flex items-center justify-center mr-2">
                              <span className="text-xs font-bold">@</span>
                            </div>
                            <p className="font-semibold text-blue-400">
                              {tweet.username !== "Unknown"
                                ? `@${tweet.username}`
                                : "Twitter User"}
                            </p>
                          </div>
                          <p className="text-gray-200">{tweet.text}</p>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          )}
        </div>
      </Template>
    </ScrollEffect>
  );
}
