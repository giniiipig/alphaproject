// src/components/Testimonials.jsx
import React, { useEffect, useMemo, useState } from "react";
import { getApiBase } from "../lib/env";

/** 후기 카드 */
function ReviewCard({ title, name, location, avatar, text, rating = 5 }) {
  return (
    <div className="bg-white p-10 rounded-2xl shadow-sm">
      {/* 제목 */}
      {title && (
        <h3 className="text-xl font-semibold text-neutral-900 mb-4">
          {title}
        </h3>
      )}

      <div className="flex items-center mb-6">
        {avatar ? (
          <img
            src={avatar}
            alt={name}
            className="w-16 h-16 rounded-full mr-4 object-cover"
          />
        ) : (
          <div className="w-16 h-16 rounded-full mr-4 bg-neutral-200" />
        )}
        <div>
          {/* 닉네임 / 작성자 */}
          <h4 className="text-lg text-neutral-800">
            {name || "익명"}
          </h4>
          {/* 위치(선택) */}
          {location && (
            <p className="text-neutral-600">{location}</p>
          )}
        </div>
      </div>
      <p className="text-lg text-neutral-700 leading-relaxed mb-6">
        {text}
      </p>
      <div className="flex text-neutral-400">
        {Array.from({ length: 5 }).map((_, i) => (
          <svg
            key={i}
            className={`w-5 h-5 mr-1 ${
              i < rating ? "text-yellow-400" : "text-neutral-300"
            }`}
            viewBox="0 0 576 512"
            aria-hidden
            fill="currentColor"
          >
            <path d="M316.9 18c-10.3-20.6-41.6-20.6-51.9 0L195 150.3 51.4 171.5c-22.9 3.4-32.1 31.5-15.5 47.7L137.8 329 113.2 474.7c-3.9 23 20.2 40.5 41 29.6L288 436l133.8 68.3c20.8 10.9 44.9-6.6 41-29.6L438.2 329 540.1 219.2c16.6-16.2 7.4-44.3-15.5-47.7L381 150.3 316.9 18z" />
          </svg>
        ))}
      </div>
    </div>
  );
}

/** 로딩용 스켈레톤 */
function ReviewSkeleton() {
  return (
    <div className="bg-white p-10 rounded-2xl shadow-sm animate-pulse">
      <div className="h-5 bg-neutral-200 rounded w-40 mb-4" />
      <div className="flex items-center mb-6">
        <div className="w-16 h-16 rounded-full bg-neutral-200 mr-4" />
        <div className="flex-1">
          <div className="h-4 bg-neutral-200 rounded w-32 mb-2" />
          <div className="h-4 bg-neutral-200 rounded w-40" />
        </div>
      </div>
      <div className="h-4 bg-neutral-200 rounded w-full mb-2" />
      <div className="h-4 bg-neutral-200 rounded w-5/6 mb-2" />
      <div className="h-4 bg-neutral-200 rounded w-4/6" />
    </div>
  );
}

/**
 * 홈 화면 후기 섹션
 */
export default function Testimonials({
  limit = 6,
  pollMs = 30000,
  enableSSE = false,
  showCTA = true,
}) {
  const BASE = useMemo(getApiBase, []);
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  async function fetchReviews(signal) {
    try {
      setErr("");

      // ✅ /api/posts 에서 가져오도록 수정
      const url = `${BASE}/posts?limit=${limit}`;
      const res = await fetch(url, {
        signal,
        credentials: "include",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const arr = Array.isArray(data)
        ? data
        : data.items || data.posts || data.results || [];

      setReviews(arr.slice(0, limit));
    } catch (e) {
      if (e.name !== "AbortError") {
        setErr("후기를 불러오지 못했어요.");
      }
    } finally {
      setLoading(false);
    }
  }

  // 최초 로드 + 폴링
  useEffect(() => {
    const ctrl = new AbortController();
    fetchReviews(ctrl.signal);
    const id = pollMs
      ? setInterval(() => fetchReviews(ctrl.signal), pollMs)
      : null;
    return () => {
      ctrl.abort();
      if (id) clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [BASE, limit, pollMs]);

  // 새 리뷰 등록 이벤트를 받으면 즉시 반영
  useEffect(() => {
    const onCreated = (ev) => {
      const newRev = ev?.detail;
      if (newRev) {
        setReviews((prev) => [newRev, ...prev].slice(0, limit));
      } else {
        const ctrl = new AbortController();
        fetchReviews(ctrl.signal);
      }
    };
    window.addEventListener("review:created", onCreated);
    return () => window.removeEventListener("review:created", onCreated);
  }, [limit]);

  // (옵션) SSE – 백엔드에 /reviews/stream 있을 때만
  useEffect(() => {
    if (!enableSSE || !BASE) return;
    let es;
    try {
      es = new EventSource(`${BASE}/reviews/stream`, {
        withCredentials: true,
      });
      es.onmessage = (ev) => {
        try {
          const rev = JSON.parse(ev.data);
          setReviews((prev) => [rev, ...prev].slice(0, limit));
        } catch {
          /* ignore */
        }
      };
      es.onerror = () => es && es.close();
    } catch {
      /* ignore */
    }
    return () => es && es.close();
  }, [BASE, enableSSE, limit]);

  // 로딩 상태
  if (loading) {
    return (
      <div className="grid lg:grid-cols-2 gap-12">
        {Array.from({ length: 2 }).map((_, i) => (
          <ReviewSkeleton key={i} />
        ))}
      </div>
    );
  }

  // 에러 상태
  if (err) {
    return (
      <div className="text-center text-neutral-500">
        {err} <span className="ml-2">잠시 후 다시 시도해주세요.</span>
      </div>
    );
  }

  // 데이터 없음
  if (!reviews.length) {
    return (
      <div className="text-center text-neutral-500">
        아직 후기가 없어요.{" "}
        <span className="text-neutral-800 font-medium">
          첫 후기의 주인공
        </span>
        이 되어주세요!
        {showCTA && (
          <div className="mt-6">
            <a
              href="/reviews/new"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-300 text-neutral-800 hover:bg-neutral-100"
            >
              ✍️ 리뷰 작성하러 가기
            </a>
          </div>
        )}
      </div>
    );
  }

  // 정상 표시
  return (
    <>
      <div className="grid lg:grid-cols-2 gap-12">
        {reviews.map((r) => (
          <ReviewCard
            key={
              r.id ||
              `${r.title || r.name}-${r.created_at || r.createdAt || Math.random()}`
            }
            title={r.title}
            // 닉네임 우선, 없으면 백엔드 author
            name={r.name || r.author || "익명"}
            location={r.location || r.addr || ""}
            avatar={r.avatar || r.image || ""}
            text={r.text || r.content || ""}
            rating={r.rating ?? 5}
          />
        ))}
      </div>

      {showCTA && (
        <div className="mt-8 text-center">
          <a
            href="/reviews/new"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-300 text-neutral-800 hover:bg-neutral-100"
          >
            ✍️ 리뷰 작성하러 가기
          </a>
        </div>
      )}
    </>
  );
}
