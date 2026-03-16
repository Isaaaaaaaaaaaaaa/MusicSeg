/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Space Grotesk', 'sans-serif'],
        mono: ['Syncopate', 'sans-serif'],
      },
      colors: {
        cyber: {
          black: '#050505',
          dark: '#0a0a0a',
          panel: '#121212',
          primary: '#00f0ff',
          secondary: '#7000ff',
          text: '#e0e0e0',
          dim: '#505050',
          // Light mode specific
          light: '#f8fafc',
          lightPanel: '#ffffff',
          lightText: '#334155',
          lightDim: '#94a3b8'
        }
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'progress-indeterminate': 'progress-indeterminate 1.5s infinite linear',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'progress-indeterminate': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' }
        }
      }
    },
  },
  plugins: [],
}
