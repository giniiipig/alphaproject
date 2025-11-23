// src/lib/api.js
import axios from "axios";
import { getApiBase } from "./env";

const baseURL = getApiBase();

export const api = axios.create({
  baseURL,
  withCredentials: true, // 세션 쿠키 자동 첨부
});

// CSRF 토큰을 쿠키에서 읽어 헤더로 첨부 (Django 기본 쿠키명: csrftoken)
api.interceptors.request.use((config) => {
  const m = document.cookie.match(/(^|;\s*)csrftoken=([^;]+)/);
  if (m) config.headers["X-CSRFToken"] = decodeURIComponent(m[2]);
  return config;
});

export default api;
