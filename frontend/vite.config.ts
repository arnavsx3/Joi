import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    fs: { allow: ['..'] },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
        headers: { Origin: 'http://127.0.0.1:7860' },
      },
    },
  },
  test: {
    environment: 'jsdom',
    clearMocks: true,
    restoreMocks: true,
  },
})
