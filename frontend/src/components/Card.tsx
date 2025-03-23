"use client";
import { cn } from "@/app/lib/utils";
import React from "react";

interface CardProps {
  title: string;
  icon: React.ReactNode;
  description: string;
  className?: string;
  direction?: string;
}

export function CardDemo({ title, icon, description, className }: CardProps) {
  return (
    <div className="max-w-xs w-full flex items-center justify-center group/card pt-16">
      <div
        className={cn(
          " cursor-pointer overflow-hidden relative card h-[250px] w-full rounded-md shadow-xl max-w-sm mx-auto backgroundImage flex flex-col justify-center p-4",
          " bg-cover",
          "border border-transparent",
          "before:absolute before:inset-0 before:rounded-md before:border before:border-white/10",
          "after:absolute after:inset-0 after:rounded-md after:border after:border-white/20 after:opacity-0",
          "hover:after:opacity-100 after:transition-opacity after:duration-300",
          className
        )}
      >
        <div className="absolute w-full h-full transition duration-300  opacity-60"></div>
        <div className="flex items-center justify-center z-10">{icon}</div>
        <div className=" text content flex flex-col space-y-2 items-center justify-center pt-8">
          <h1 className="font-bold text-lg md:text-xl text-gray-50 relative z-10">
            {title}
          </h1>
          <p className=" text-center font-semibold pt-4 text-md text-gray-500">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
}
