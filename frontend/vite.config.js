import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'https://localhost:9001',
        changeOrigin: true,
        secure: false
      },
      '/photos': {
        target: 'https://localhost:9001',
        changeOrigin: true,
        secure: false
      },
      '/captures': {
        target: 'https://localhost:9001',
        changeOrigin: true,
        secure: false
      },
      '/models': {
        target: 'https://localhost:9001',
        changeOrigin: true,
        secure: false
      }
    }
  }
});
