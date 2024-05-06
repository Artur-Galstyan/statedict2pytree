/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./torch2jax/**/*.{html,js}"],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui")],
};
