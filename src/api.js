// src/api.js
import axios from "axios";
import { getApiBase } from "./lib/env";

const baseURL = getApiBase();

export const api = axios.create({
  baseURL,               // ✅ 백엔드 절대주소 혹은 /api 프록시
  withCredentials: true, // ✅ 세션 쿠키 포함
});

// (선택) CSRF 쿠키를 헤더로
api.interceptors.request.use((config) => {
  const m = document.cookie.match(/(^|;\s*)csrftoken=([^;]+)/);
  if (m) config.headers["X-CSRFToken"] = decodeURIComponent(m[2]);
  return config;
});
