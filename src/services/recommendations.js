// src/services/recommendations.js

import { getApiBase } from "../lib/env";

const BASE = getApiBase();

/** 공통 fetch 유틸 (JSON 파싱·에러 메시지 보강·credentials 포함) */
async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(opts.headers || {}),
    },
  });

  let data = null;
  const text = await res.text().catch(() => "");
  try { data = text ? JSON.parse(text) : null; } catch { /* ignore */ }

  if (!res.ok) {
    const msg =
      (data && (data.message || data.detail || data.error)) ||
      (text || `HTTP ${res.status}`);
    throw new Error(typeof msg === "string" ? msg : "Request failed");
  }
  return data;
}

/** 저장된(내) 여행 코스 목록 */
export async function getSavedTrips({ signal } = {}) {
  return req("/me/trips", { method: "GET", signal });
}

/** '가장 최근 추천'을 서버에 저장 (PlanTrip에서 성공 직후 호출) */
export async function saveLatestRecommendation(body = { latest: true }, { signal } = {}) {
  // 기존 구현은 /api/recommendations/save 를 사용하고 있었음.
  // 백엔드가 다른 경로를 원하면 아래 path만 바꿔주면 됩니다.
  return req("/recommendations/save", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}
