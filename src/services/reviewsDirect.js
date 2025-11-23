// src/services/reviewsDirect.js
import axios from "axios";
import { getAiBase } from "../lib/env";

const BASE = getAiBase();
const http = axios.create({ baseURL: BASE, timeout: 60_000 });

/**
 * AI 서버에 직접 리뷰 생성(백엔드 우회)
 * - images: 문자열 URL 또는 dataURL(base64) 배열 지원
 * - 서버가 /api/v1/reviews/create 또는 /api/v1/reviews 로 받을 수 있도록 이중 시도
 */
export async function createReviewDirect({ title, content, rating = 5, images = [], author = "익명", location = null }) {
  const payload = {
    title: String(title || "").trim(),
    content: String(content || "").trim(),
    rating: Math.max(1, Math.min(5, Number(rating || 5))),
    images,                 // 표준 키
    image_urls: images,     // 별칭 호환
    author,
    author_name: author,    // 별칭 호환
    location,
    via: "frontend-direct"
  };
  if (!payload.title || !payload.content) throw new Error("제목/내용은 필수입니다.");

  // 1차: /api/v1/reviews/create
  try {
    const { data } = await http.post("/api/v1/reviews/create", payload);
    return data;
  } catch (e1) {
    // 2차: /api/v1/reviews
    if (e1?.response?.status === 404) {
      const { data } = await http.post("/api/v1/reviews", payload);
      return data;
    }
    throw e1;
  }
}
