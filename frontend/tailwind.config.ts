import type { Config } from "tailwindcss";
const { nextui } = require("@nextui-org/react");

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./node_modules/@nextui-org/theme/dist/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      colors: {
        "dark-100": "#151515",
        "dark-200": "#090909",
        "dark-300": "#161616",
        "dark-50": "#191919",
        "dark-150": "#212121",
      },
      fontFamily: {
        inter: ["var(--font-inter)"],
        lora: ["var(--font-lora)"],
        roboto: ["var(--font-roboto)"],
      },

      fontSize: {
        custom: [
          "15px",
          {
            lineHeight: "28px",
          },
        ],
        "custom-xs": [
          "13px",
          {
            lineHeight: "21px",
          },
        ],
        "custom-sm": "15px",
      },
    },
    boxShadow: {
      custom_focus: "0 0 10px rgba(0, 0, 0, 0.15)", // Custom shadow
      custom_unfocus: "0 0 10px rgba(0, 0, 0, 0.12)", // Custom shadow
    },
  },
  darkMode: "class",
  plugins: [nextui(), require("@tailwindcss/typography")],
};
export default config;
