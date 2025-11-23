// src/lib/env.js
const DEFAULT_API_BASE = "/api";
const DEFAULT_AI_BASE = "/ai";

const hasWindow = typeof window !== "undefined";
const hasDocument = typeof document !== "undefined";
const hasImportMeta = typeof import.meta !== "undefined" && !!import.meta?.env;

function normalizeBase(value, fallback) {
  if (value === undefined || value === null) return fallback;
  const str = String(value).trim();
  if (!str) return fallback;
  if (str === "/") return "/";
  return str.replace(/\/+$/, "") || fallback;
}

function readWindow(key) {
  if (!hasWindow) return null;
  return window[key] ?? null;
}

function readMeta(name) {
  if (!hasDocument) return null;
  return document.querySelector(`meta[name="${name}"]`)?.content || null;
}

function readEnv(keys = []) {
  if (!hasImportMeta) return null;
  for (const key of keys) {
    const value = import.meta.env[key];
    if (value) return value;
  }
  return null;
}

function joinUrl(base, path = "") {
  if (!path) return base;
  const prefixed = path.startsWith("/") ? path : `/${path}`;
  return base === "/" ? prefixed : `${base}${prefixed}`;
}

export function getApiBase() {
  const candidate =
    readWindow("__API_BASE") ||
    readWindow("__API_BASE__") ||
    readMeta("api-base") ||
    readEnv(["VITE_API_BASE", "VITE_API_BASE_URL", "VITE_BACKEND_ORIGIN"]);
  return normalizeBase(candidate, DEFAULT_API_BASE);
}

export function getAiBase() {
  const candidate =
    readWindow("__AI_BASE") ||
    readWindow("__AI_BASE__") ||
    readMeta("ai-base") ||
    readEnv(["VITE_AI_URL"]);
  return normalizeBase(candidate, DEFAULT_AI_BASE);
}

export function buildApiUrl(path = "") {
  return joinUrl(getApiBase(), path);
}

export function buildAiUrl(path = "") {
  return joinUrl(getAiBase(), path);
}
