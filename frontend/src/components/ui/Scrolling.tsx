"use client";
import React, { useEffect } from "react";

const SmoothScroll: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  useEffect(() => {
    (async () => {
      const LocomotiveScroll = (await import("locomotive-scroll")).default;
      const locomotiveScroll = new LocomotiveScroll();
    })();
  }, []);

  return <div>{children}</div>;
};

export default SmoothScroll;
