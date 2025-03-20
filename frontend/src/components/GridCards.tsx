"use client";
import { CardDemo } from "./Card";
import { motion, useInView } from "framer-motion";
import React, { useRef } from "react";

interface Props {
  direction?: string;
}
export function Cards({ direction }: Props) {
  const ref = useRef(null);
  const Inview = useInView(ref, { once: false });
  const dir = direction || "down";
  const text_effect = {
    show: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: { type: "spring", duration: 1.5 },
    },
    hide: { opacity: 0, x: dir === "down" ? -200 : 200, scale: 0.8 },
    transition: { type: "spring", duration: 1.5 },
  };
  const cards = [
    {
      title: "In-Depth Analysis",
      description: "Gain actionable insights with deeper analytics.",
      icon: (
        <svg
          width="80"
          height="80"
          viewBox="0 0 80 80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M19 16C19 13.7909 20.7909 12 23 12H59C61.2091 12 63 13.7909 63 16V64C63 66.2091 61.2091 68 59 68H23C20.7909 68 19 66.2091 19 64V16Z"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M21 28H17M21 20H17M21 36H17M21 52H17M21 60H17M21 44H17"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M29 24H53"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M29 32H45"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M29 40H53"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M29 48H45"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M29 56H53"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
    {
      title: "Sentimental Tracking",
      description: "Extracts and evaluates emotional tone from tweets.",
      icon: (
        <svg
          width="80"
          height="80"
          viewBox="0 0 80 80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M31.161 18.661C36.8204 16.3168 43.1792 16.3168 48.8387 18.661C54.4981 21.0052 58.9944 25.5016 61.3387 31.161C63.6829 36.8204 63.6829 43.1792 61.3387 48.8387C58.9944 54.4981 54.4981 58.9944 48.8387 61.3387C43.1792 63.6829 36.8204 63.6829 31.161 61.3387C25.5016 58.9944 21.0052 54.4981 18.661 48.8387C16.3168 43.1792 16.3168 36.8204 18.661 31.161C21.0052 25.5016 25.5016 21.0052 31.161 18.661Z"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M36.4645 31.4645C38.7283 30.5268 41.2718 30.5268 43.5356 31.4645C45.7993 32.4022 47.5979 34.2007 48.5356 36.4645C49.4732 38.7283 49.4732 41.2718 48.5356 43.5356C47.5979 45.7993 45.7993 47.5979 43.5356 48.5356C41.2718 49.4732 38.7283 49.4732 36.4645 48.5356C34.2007 47.5979 32.4022 45.7993 31.4645 43.5356C30.5268 41.2718 30.5268 38.7283 31.4645 36.4645C32.4022 34.2007 34.2007 32.4022 36.4645 31.4645Z"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M40 12L40.008 23.3702"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M56.6299 40.008L68.0001 40"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M39.9922 56.6299L40.0002 68.0001"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M23.3702 39.9922L12 40.0002"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
    {
      title: "Visualization",
      description: "Insightful sentiment trends through dynamic graphs.",
      icon: (
        <svg
          width="80"
          height="80"
          viewBox="0 0 80 80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M31 66C31 67.1046 30.1046 68 29 68H15C13.8954 68 13 67.1046 13 66V46C13 44.8954 13.8954 44 15 44H29C30.1046 44 31 44.8954 31 46"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M49 66C49 67.1046 48.1046 68 47 68H33C31.8954 68 31 67.1046 31 66V30C31 28.8954 31.8954 28 33 28H47C48.1046 28 49 28.8954 49 30"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M67 14C67 12.8954 66.1046 12 65 12H51C49.8954 12 49 12.8954 49 14V66C49 67.1046 49.8954 68 51 68H65C66.1046 68 67 67.1046 67 66V14Z"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M72 68H8"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
    {
      title: "Powered By AI",
      description: "Advanced NLP for precise sentiment classification.",
      icon: (
        <svg
          width="80"
          height="80"
          viewBox="0 0 80 80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M71.0144 39.9996C69.8841 37.582 68.4643 35.2823 66.7672 33.1542C60.271 25.0082 50.4189 20.2637 39.9998 20.2637C29.5807 20.2637 19.7286 25.0082 13.2324 33.1542C11.5348 35.2828 10.1148 37.5832 8.98438 40.0014"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M71.0144 39.9995C69.884 42.4177 68.4639 44.7181 66.7664 46.8467C60.2702 54.9927 50.4181 59.7372 39.9989 59.7372C29.5798 59.7372 19.7277 54.9927 13.2315 46.8467C11.5344 44.7186 10.1146 42.4189 8.98438 40.0013"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M35.3092 49.7415C38.2736 51.1691 41.7268 51.1691 44.6912 49.7415C47.6556 48.3139 49.8086 45.6142 50.5408 42.4064C51.2729 39.1987 50.5045 35.8321 48.4531 33.2597C46.4016 30.6872 43.2904 29.189 40.0002 29.189C36.71 29.189 33.5988 30.6872 31.5473 33.2597C29.4959 35.8321 28.7275 39.1987 29.4596 42.4064C30.1918 45.6142 32.3448 48.3139 35.3092 49.7415Z"
            stroke="#C2CCDE"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
  ];
  return (
    <>
      <motion.div
        ref={ref}
        initial="hide"
        animate={Inview ? "show" : "hide"}
        variants={text_effect}
        className="flex items-center flex-col text-center justify-center"
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
      </motion.div>
      <div className="container flex justify-center mx-auto py-12 mb-56 ">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {cards.map((card, index) => (
            <CardDemo key={index} {...card} />
          ))}
        </div>
      </div>
    </>
  );
}
