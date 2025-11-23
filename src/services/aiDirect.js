// src/services/aiDirect.js
import axios from "axios";
import { getAiBase } from "../lib/env";

// 1) .env에 VITE_AI_URL 있으면 그걸 쓰고
// 2) 없으면 dev 서버의 프록시 경로(/ai)를 기본으로 사용
const BASE = getAiBase();

export const ai = axios.create({
  baseURL: BASE,
  timeout: 60000,
});

// PlanTrip payload → 텍스트 프롬프트
function toPrefText(p) {
  const style1 = p.theme?.[0] === "outdoor" ? "실외" : "실내";
  const style2 = p.theme?.[1] === "active" ? "활동적" : "휴식형";
  return [
    p.region,
    p.startDate && p.endDate && `${p.startDate}~${p.endDate}`,
    p.transport && `이동수단:${p.transport}`,
    `${style1}/${style2}`,
    p.budget ? `예산:${p.budget}원` : "",
    p.notes && `선호:${p.notes}`,
  ]
    .filter(Boolean)
    .join(", ");
}

// 좌표/필드 정규화 (서버 응답 키가 달라도 안전)
function normalizeItem(it, i) {
  const lat = Number(
    it.lat ??
      it.latitude ??
      it.coord?.lat ??
      it.coords?.lat ??
      it.y ??
      it.mapy
  );
  const lon = Number(
    it.lon ??
      it.lng ??
      it.longitude ??
      it.coord?.lon ??
      it.coords?.lon ??
      it.x ??
      it.mapx
  );
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return {
    title: it.title || it.name || `장소 ${i + 1}`,
    overview: it.overview || it.desc || "",
    image_url: it.image_url || it.imageUrl || "",
    addr: it.addr || it.address || "",
    lat,
    lon,
    _raw: it, // 디버그용 원본 보존
  };
}

export async function fetchTopK(payload) {
  const body = {
    mode: "text",
    pref_text: toPrefText(payload),
    current_location: { lat: 37.5665, lon: 126.9780 },
    options: {
      top_k: payload.topk || 5,
      algo: "hybrid",
    },
  };

  const { data } = await ai.post(`/api/v1/recommend/top5`, body);

  // 서버가 배열 혹은 {items: [...]} 둘 다 수용
  const arr = Array.isArray(data)
    ? data
    : Array.isArray(data?.items)
    ? data.items
    : [];
  return arr.map(normalizeItem).filter(Boolean);
}
