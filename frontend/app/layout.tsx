import type { Metadata } from "next";
import "@/public/globals.css";

import { Lora } from "next/font/google";
import { Inter } from "next/font/google";
import { Roboto } from "next/font/google";
// const inter = Roboto({ subsets: ["latin"], weight: "400" });

export const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
  style: ["normal"],
  weight: ["400", "500", "600", "700"],
});

export const lora = Lora({
  subsets: ["latin"],
  display: "swap",
  style: ["normal", "italic"],
  variable: "--font-lora",
  weight: ["400", "500", "600", "700"],
});

export const roboto = Roboto({
  subsets: ["latin"],
  display: "swap",
  style: ["normal", "italic"],
  variable: "--font-roboto",
  weight: ["100", "300", "400", "500", "700", "900"],
});

export const metadata: Metadata = {
  title: "IncarnaMind",
  description: "Your knowledge, incarnate",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${lora.variable} ${roboto.variable} font-roboto`}
      >
        {children}
      </body>
    </html>
  );
}
