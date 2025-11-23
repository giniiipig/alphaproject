// src/services/ai.js
import axios from "axios";
import { getAiBase } from "../lib/env";

const BASE = getAiBase();
export const ai = axios.create({ baseURL: BASE, timeout: 60_000 });

/**
 * 텍스트 기반 Top-5 추천
 */
export async function getRecommendations(
  text,
  location = { lat: 37.5665, lon: 126.9780 },
  options = { top_k: 5, algo: "hybrid" }
) {
  const { data } = await ai.post("/api/v1/recommend/top5", {
    mode: "text",
    pref_text: text,
    current_location: location,
    options,
  });
  return data?.items ?? [];
}

/**
 * (옵션) 리뷰 문장 보정/초안 생성
 * 서버 사양에 따라 엔드포인트를 /api/v1/review/generate 또는 /api/v1/reviews/generate 로 사용
 */
export async function generateReviewDraft({ title, content, images = [], tone = "neutral" }) {
  try {
    const { data } = await ai.post("/api/v1/review/generate", { title, content, images, tone });
    return data; // { title?, content } 가정
  } catch (e1) {
    if (e1?.response?.status === 404) {
      const { data } = await ai.post("/api/v1/reviews/generate", { title, content, images, tone });
      return data;
    }
    throw e1;
  }
}
