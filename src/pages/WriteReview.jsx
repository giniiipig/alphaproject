// src/pages/WriteReview.jsx
import React, { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header.jsx";
import { createReview } from "../services/reviews";
import { createReviewDirect } from "../services/reviewsDirect";

/* ───────────────────────── 별점 컴포넌트 ───────────────────────── */
function StarRating({ value, onChange, size = 28 }) {
  const [hover, setHover] = useState(0); // 0이면 미리보기 없음
  const active = hover || value;

  return (
    <div className="flex items-center gap-2" role="radiogroup" aria-label="평점 선택">
      {Array.from({ length: 5 }).map((_, i) => {
        const idx = i + 1;
        const filled = active >= idx;
        return (
          <button
            key={idx}
            type="button"
            role="radio"
            aria-checked={value === idx}
            onMouseEnter={() => setHover(idx)}
            onMouseLeave={() => setHover(0)}
            onFocus={() => setHover(idx)}
            onBlur={() => setHover(0)}
            onClick={() => onChange(idx)}
            className="transition-transform hover:-translate-y-0.5 focus:-translate-y-0.5 focus:outline-none"
            style={{ lineHeight: 1 }}
            title={`${idx}점`}
          >
            <svg
              width={size}
              height={size}
              viewBox="0 0 24 24"
              className={filled ? "text-yellow-400" : "text-neutral-300"}
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M12 .587l3.668 7.431L24 9.748l-6 5.842 1.416 8.257L12 19.771l-7.416 4.076L6 15.59 0 9.748l8.332-1.73z" />
            </svg>
          </button>
        );
      })}
      <span className="ml-2 text-sm text-neutral-700">{value} / 5</span>
    </div>
  );
}

/* ───────────────────────── 유틸: 파일→dataURL ───────────────────────── */
function fileToDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

export default function WriteReview() {
  const navigate = useNavigate();

  // 폼 상태
  const [nickname, setNickname] = useState("");
  const [title, setTitle] = useState("");
  const [rating, setRating] = useState(5);
  const [content, setContent] = useState("");
  const [location, setLocation] = useState(null);
  const [images, setImages] = useState([]); // dataURL 또는 URL 문자열 배열
  const [directAI, setDirectAI] = useState(false); // ✅ AI 직결 토글

  // 초안 복원
  useEffect(() => {
    const raw = localStorage.getItem("reviewDraft");
    if (raw) {
      try {
        const d = JSON.parse(raw);
        setNickname(d.nickname || "");
        setTitle(d.title || "");
        setRating(d.rating ?? 5);
        setContent(d.content || "");
        setImages(Array.isArray(d.images) ? d.images : []);
      } catch {}
    }
  }, []);

  // 초안 저장
  useEffect(() => {
    localStorage.setItem(
      "reviewDraft",
      JSON.stringify({ nickname, title, rating, content, images })
    );
  }, [nickname, title, rating, content, images]);

  // 이미지 추가/삭제
  async function handleFiles(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const urls = await Promise.all(files.map(fileToDataURL));
    setImages((prev) => [...prev, ...urls]);
  }
  function removeImage(idx) {
    setImages((prev) => prev.filter((_, i) => i !== idx));
  }

  // 제출
  const { mutate, isPending, isError, error } = useMutation({
    mutationFn: async (body) => {
      if (directAI) {
        const data = await createReviewDirect(body);
        try {
          window.dispatchEvent(new CustomEvent("review:created", { detail: data }));
        } catch {}
        return data;
      }
      return createReview(body);
    },
    onSuccess: () => {
      localStorage.removeItem("reviewDraft");
      navigate("/", { replace: true });
    },
  });

  const onSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) {
      alert("제목과 내용을 입력해주세요.");
      return;
    }
    mutate({ title, rating, content, location, images, author: nickname || "익명" });
  };

  const inputClass =
    "h-12 w-full border border-[#D4D4D4] rounded-lg px-3 placeholder:text-[#ADAEBC]";
  const labelClass = "block text-sm text-[#404040] mb-2";

  return (
    <div className="min-h-screen bg-[#FAFAFA]">
      <Header />
      <main className="w-full">
        <div className="max-w-[1152px] mx-auto pt-12 pb-16 px-4">
          <div className="text-center mb-8">
            <h1 className="text-[32px] md:text-[36px] leading-[40px] text-[#262626]">리뷰 작성</h1>
            <p className="text-[16px] md:text-[18px] leading-[28px] text-[#525252] mt-2">
              방문하신 장소에 대한 솔직한 후기를 남겨주세요
            </p>
          </div>

          <div className="bg-white border border-[#E5E5E5] shadow-sm rounded-2xl p-6 md:p-8">
            <form onSubmit={onSubmit} className="space-y-6">
              {/* 닉네임 */}
              <section>
                <label className={labelClass}>닉네임 (선택)</label>
                <input
                  className={inputClass}
                  value={nickname}
                  onChange={(e) => setNickname(e.target.value)}
                  placeholder="로그인 없이 작성 시 표시될 이름 (미입력 시 '익명')"
                />
              </section>

              {/* 제목 */}
              <section>
                <label className={labelClass}>제목</label>
                <input
                  className={inputClass}
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="리뷰 제목을 입력하세요"
                />
              </section>

              {/* 별점 */}
              <section>
                <label className={labelClass}>평점</label>
                <StarRating value={rating} onChange={setRating} />
              </section>

              {/* 내용 */}
              <section>
                <label className={labelClass}>내용</label>
                <textarea
                  className="w-full border border-[#D4D4D4] rounded-lg p-3 min-h-[180px] text-sm placeholder:text-[#ADAEBC]"
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="내용을 입력하세요"
                />
              </section>

              {/* ✅ 이미지 업로드 & 미리보기 */}
              <section>
                <label className={labelClass}>이미지 (여러 장 가능)</label>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFiles}
                  className="block w-full text-sm"
                />
                {!!images.length && (
                  <div className="mt-3 grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-3">
                    {images.map((src, i) => (
                      <div key={i} className="relative">
                        <img
                          src={src}
                          alt={`리뷰 이미지 ${i + 1}`}
                          className="w-full h-24 object-cover rounded-lg border"
                        />
                        <button
                          type="button"
                          onClick={() => removeImage(i)}
                          className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-black/70 text-white text-xs"
                          aria-label="이미지 삭제"
                          title="삭제"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              {/* ✅ 전송 경로 토글 */}
              <section className="flex items-center gap-2">
                <input
                  id="direct-ai"
                  type="checkbox"
                  checked={directAI}
                  onChange={(e) => setDirectAI(e.target.checked)}
                />
                <label htmlFor="direct-ai" className="text-sm text-neutral-700">
                  백엔드 거치지 않고 AI 서버로 바로 전송
                </label>
              </section>

              {/* 제출 */}
              <section className="pt-2">
                <button
                  type="submit"
                  disabled={isPending}
                  className="w-full h-11 bg-[#262626] text-white rounded-lg hover:bg-black disabled:opacity-60"
                >
                  {isPending ? "등록 중…" : "리뷰 등록"}
                </button>
                {isError && (
                  <div className="text-sm text-red-600 mt-2">
                    {error?.message || "리뷰 등록에 실패했습니다."}
                  </div>
                )}
              </section>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}
