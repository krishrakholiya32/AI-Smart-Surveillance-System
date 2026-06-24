/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg:      '#0a0c10',
          surface: '#0d1117',
          border:  '#1e3a4a',
          cyan:    '#00e5ff',
          green:   '#00ff99',
          red:     '#ff4b4b',
          orange:  '#ffaa00',
          blue:    '#4a9eff',
          muted:   '#5a8fa8',
          dim:     '#8ab4c8',
        },
      },
      fontFamily: {
        mono: ['"Share Tech Mono"', 'monospace'],
        sans: ['Rajdhani', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
