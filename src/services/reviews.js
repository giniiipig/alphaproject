// src/services/reviews.js
import { buildApiUrl } from "../lib/env";

// 숫자 범위 제한
function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

// 공통 POST 유틸
async function postJson(path, body, { withCreds = true } = {}) {
  const url =
    typeof path === "string" && /^https?:\/\//.test(path)
      ? path
      : buildApiUrl(path);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: withCreds ? "include" : "omit",
    body: JSON.stringify(body),
  });

  const text = await res.text().catch(() => "");
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* ignore JSON parse error */
  }

  if (!res.ok) {
    const msg =
      (data && (data.message || data.detail || data.error)) ||
      text ||
      `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }

  return data;
}

/**
 * 최신 리뷰 목록 (지금은 Post 목록을 리뷰처럼 사용)
 * 백엔드에서 /api/posts/ 응답을 다음 정도로 가정:
 *   [
 *     { id, title, content, author, created_at, ... },
 *     ...
 *   ]
 */
export async function listLatestReviews(limit = 6) {
  const res = await fetch(buildApiUrl(`/posts?limit=${limit}`), {
    credentials: "include",
  });

  if (!res.ok) {
    throw new Error(`리뷰 목록을 불러오지 못했습니다. (HTTP ${res.status})`);
  }

  const data = await res.json();
  const arr = Array.isArray(data)
    ? data
    : data.items || data.posts || data.results || [];

  return arr.slice(0, limit).map((p) => ({
    id: p.id,
    title: p.title,
    text: p.content,
    content: p.content,
    name: p.author || "익명",
    author: p.author || "익명",
    rating: 5,
    location: p.location || null,
    createdAt: p.created_at,
  }));
}

/**
 * (옵션) 내 리뷰 – 필요 없으면 안 써도 됨
 */
export async function getMyReviews() {
  const res = await fetch(buildApiUrl(`/posts`), {
    credentials: "include",
  });
  if (!res.ok) {
    throw new Error(`내 리뷰를 불러오지 못했습니다. (HTTP ${res.status})`);
  }
  return res.json();
}

/**
 * 리뷰 작성
 * - WriteReview.jsx 에서 호출
 * - 백엔드의 /api/posts/create/에 게시글을 생성하고
 * - 프론트에서 쓰기 편한 리뷰 객체로 변환해서 반환
 */
export async function createReview({
  title,
  rating,
  content,
  location = null,
  images = [],
  author, // 닉네임 (WriteReview에서 넘겨주는 값)
}) {
  const r = clamp(Number(rating || 5), 1, 5);

  const base = {
    title: String(title || "").trim(),
    content: String(content || "").trim(),
    rating: r,
    location,
    images: Array.isArray(images) ? images : [],
  };

  if (!base.title || !base.content) {
    throw new Error("제목과 내용을 입력해주세요.");
  }

  // ✅ 백엔드 Alpha 게시판 API 스펙에 맞춘 payload
  // 예시: { title: "...", content: "...", author_id: 1 }
  const payload = {
    title: base.title,
    content: base.content,
    // TODO: 나중에 실제 로그인 유저 ID로 교체
    author_id: 1,
  };

  // 세션 로그인 안 써도 되면 withCreds: false 여도 됨
  const data = await postJson("/posts/create/", payload, {
    withCreds: false,
  });

  // ✅ 프론트에서 사용할 "리뷰" 형태로 정규화
  const review = {
    id: data.id,
    title: data.title,
    content: data.content,
    text: data.content,
    rating: r,
    location: base.location,
    images: base.images,
    // 닉네임을 name으로 넣어준다 → Testimonials에서 r.name으로 사용
    name: author || "익명",
    // author 필드도 같이 넣어주기
    author: author || data.author || "익명",
    createdAt: data.created_at,
  };

  return review;
}
