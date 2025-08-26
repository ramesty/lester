import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '../frontend/node_modules/@tailwindcss/vite/dist/index.d.mts'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
})
