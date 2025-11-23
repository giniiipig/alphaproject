// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = "http://100.107.38.99:8000";      // Django
const aiServer = "http://100.125.47.102:8000";    // FastAPI (AI)

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": {
        target: backend,
        changeOrigin: true,
        secure: false,
      },
      "/auth": {
        target: backend,
        changeOrigin: true,
        secure: false,
      },
      // 🔥 여기 수정
      "/ai": {
        target: aiServer,
        changeOrigin: true,
        secure: false,
        // '/ai/api/...' -> '/api/...' 로 잘라서 AI 서버로 보냄
        rewrite: (path) => path.replace(/^\/ai/, ""),
      },
    },
  },
});
