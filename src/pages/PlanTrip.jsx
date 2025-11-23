// src/pages/PlanTrip.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header.jsx";
import { useMutation } from "@tanstack/react-query";
import { fetchTopK } from "../services/aiDirect"; // AI 직결 호출

const fmt = (d) => new Date(d).toISOString().slice(0, 10);
const today = new Date();
const twoDaysLater = new Date(+today + 2 * 86400000);

export default function PlanTrip() {
  const navigate = useNavigate();

  // ---- form state ----
  const [startDate, setStartDate] = useState(fmt(today));
  const [endDate, setEndDate] = useState(fmt(twoDaysLater));
  const [region, setRegion] = useState("제주특별자치도");
  const [inout, setInout] = useState("실외");          // "실내" | "실외"
  const [activity, setActivity] = useState("휴식형");   // "휴식형" | "활동적"
  const [customStyle, setCustomStyle] = useState("");
  const [transport, setTransport] = useState("자동차"); // "도보" | "대중교통" | "자동차" | "자전거"
  const [budget, setBudget] = useState("");

  const dayCount = useMemo(() => {
    const a = new Date(startDate), b = new Date(endDate);
    return Math.max(1, Math.round((+b - +a) / 86400000) + 1);
  }, [startDate, endDate]);

  // 최근 검색
  const [recent, setRecent] = useState([]);
  useEffect(() => {
    try {
      const r = JSON.parse(localStorage.getItem("recentSearches") || "[]");
      setRecent(Array.isArray(r) ? r : []);
    } catch { setRecent([]); }
  }, []);

  // AI 직결 추천 뮤테이션
  const aiMut = useMutation({
    mutationFn: fetchTopK,
    onSuccess: (items) => {
      // 지도 페이지에서 사용할 결과 저장
      // (최신 키 + 하위호환 키 둘 다 저장)
      localStorage.setItem("latest_ai_items", JSON.stringify(items));
      localStorage.setItem("ai_result", JSON.stringify(items));
      navigate("/map");
    },
    onError: (e) => {
      const msg = e?.response?.data?.message || e?.message || "AI 추천 실패";
      alert(msg);
    },
  });

  const onSubmit = (e) => {
    e.preventDefault();

    const prefs = {
      startDate, endDate, region,
      transport,
      // 메타 표시에 쓰는 보기 좋은 문자열
      theme: `${inout} • ${activity}`,
      customStyle: (customStyle || "").trim(),
      budget: budget ? Number(budget) : null,
    };

    // 1) 다음 페이지에서 사용하도록 저장
    localStorage.setItem("tripPreferences", JSON.stringify(prefs));

    // 2) 최근 검색 업데이트(최대 6개)
    const card = {
      period: `${startDate} - ${endDate}`,
      title: `${region} ${dayCount}일`,
      subtitle: `${activity} • ${transport} • ${inout}`,
      createdAt: Date.now(),
    };
    const next = [card, ...recent].slice(0, 6);
    localStorage.setItem("recentSearches", JSON.stringify(next));
    setRecent(next);

    // 3) AI 서버로 직접 추천 요청
    aiMut.mutate({
      region,
      startDate,
      endDate,
      transport,
      // AI 서버용 테마는 배열로 전달 (실외/활동적 → outdoor/active 매핑)
      theme: [
        inout === "실외" ? "outdoor" : "indoor",
        activity === "활동적" ? "active" : "relax",
      ],
      notes: prefs.customStyle,
      budget: prefs.budget,
      // ✅ 일수×5, 최소 5
      topk: Math.max(5, dayCount * 5),
    });
  };

  // 공통 버튼 스타일
  const segBtn = (active) =>
    `h-10 rounded-lg border text-sm px-4 ${
      active ? "bg-black text-white border-black" : "border-neutral-300 text-neutral-700 hover:bg-neutral-50"
    }`;

  return (
    <div className="relative w-[1440px] min-h-[1779px] mx-auto bg-white border-2 border-gray-300 rounded-lg">
      {/* 공용 헤더 */}
      <Header />

      {/* main */}
      <main className="absolute left-0 top-[73px] w-full bg-[#FAFAFA]">
        <div className="w-[1152px] mx-auto pt-12 pb-12">
          {/* title */}
          <div className="text-center mb-8">
            <h1 className="text-[36px] leading-[40px] text-[#262626]">여행 계획 만들기</h1>
            <p className="text-[18px] leading-[28px] text-[#525252] mt-2">
              원하는 여행 조건을 입력하면 맞춤형 여행 코스를 추천해드립니다
            </p>
          </div>

          {/* form card */}
          <div className="bg-white border border-[#E5E5E5] shadow-sm rounded-2xl p-8">
            <form className="space-y-6" onSubmit={onSubmit}>
              {/* 날짜 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm text-[#404040] mb-2">여행 시작일</label>
                  <input type="date" value={startDate} onChange={(e)=>setStartDate(e.target.value)}
                         className="h-12 w-full border border-[#D4D4D4] rounded-lg px-3" />
                </div>
                <div>
                  <label className="block text-sm text-[#404040] mb-2">여행 종료일</label>
                  <input type="date" value={endDate} onChange={(e)=>setEndDate(e.target.value)}
                         min={startDate}
                         className="h-12 w-full border border-[#D4D4D4] rounded-lg px-3" />
                </div>
              </div>

              {/* 지역 */}
              <div>
                <label className="block text-sm text-[#404040] mb-2">여행 지역</label>
                <select value={region} onChange={(e)=>setRegion(e.target.value)}
                        className="h-12 w-full border border-[#D4D4D4] rounded-lg px-3">
                  <option>서울특별시</option>
                  <option>부산광역시</option>
                </select>
              </div>

              {/* 여행 스타일 */}
              <div>
                <label className="block text-sm text-[#404040] mb-2">여행 스타일</label>

                <div className="border border-[#E5E5E5] rounded-lg p-3 mb-3 flex items-center justify-end gap-3">
                  <span className="text-sm text-neutral-700">실내</span>
                  <button type="button" className={segBtn(inout==="실외")} onClick={()=>setInout(inout==="실외"?"실내":"실외")}>
                    {inout}
                  </button>
                  <span className="text-sm text-neutral-700">실외</span>
                </div>

                <div className="border border-[#E5E5E5] rounded-lg p-3 mb-3 flex items-center justify-end gap-3">
                  <span className="text-sm text-neutral-700">휴식형</span>
                  <button type="button" className={segBtn(activity==="활동적")} onClick={()=>setActivity(activity==="활동적"?"휴식형":"활동적")}>
                    {activity}
                  </button>
                  <span className="text-sm text-neutral-700">활동적</span>
                </div>

                <div className="border border-[#E5E5E5] rounded-lg p-4">
                  <div className="text-sm text-[#404040] mb-2">나만의 여행 스타일</div>
                  <textarea
                    value={customStyle}
                    onChange={(e)=>setCustomStyle(e.target.value)}
                    rows={4}
                    placeholder="선호하는 여행 스타일을 자유롭게 입력해주세요 (예: 맛집 탐방, 사진 촬영, 역사 탐방, 자연 감상 등)"
                    className="w-full border border-[#D4D4D4] rounded-lg p-3 text-sm placeholder:text-[#ADAEBC]"
                  />
                  <div className="text-xs text-[#737373] mt-2">
                    개인 취향을 입력하면 더 맞춤화된 추천을 받을 수 있습니다
                  </div>
                </div>
              </div>

              {/* 이동 수단 */}
              <div>
                <label className="block text-sm text-[#404040] mb-2">이동 수단</label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {["도보","대중교통","자동차","자전거"].map((t)=>(
                    <button
                      key={t}
                      type="button"
                      onClick={()=>setTransport(t)}
                      className={`h-18 py-5 rounded-lg border ${transport===t ? "bg-black text-white border-black" : "border-[#E5E5E5] text-[#404040]"}`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              {/* 예산 */}
              <div>
                <label className="block text-sm text-[#404040] mb-2">예산 (선택사항)</label>
                <input
                  type="number"
                  min="0"
                  placeholder="예산을 입력하세요 (원)"
                  value={budget}
                  onChange={(e)=>setBudget(e.target.value)}
                  className="h-12 w-full border border-[#D4D4D4] rounded-lg px-3 placeholder:text-[#ADAEBC]"
                />
                <div className="text-xs text-[#737373] mt-2">
                  예산을 입력하지 않으면 다양한 가격대의 장소를 추천해드립니다
                </div>
              </div>

              {/* 제출 */}
              <div className="pt-2">
                <button
                  type="submit"
                  className="w-full h-11 bg-[#262626] text-white rounded-lg hover:bg-black disabled:opacity-60"
                  disabled={aiMut.isPending}
                >
                  {aiMut.isPending ? "AI 계산 중..." : "여행 코스 추천 받기"}
                </button>
              </div>
            </form>
          </div>

          {/* 최근 검색 */}
          <section className="mt-12">
            <h3 className="text-[20px] text-[#262626] mb-4">최근 검색</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {recent.slice(0,3).map((r, idx)=>(
                <div key={idx} className="bg-white border border-[#E5E5E5] rounded-xl p-5">
                  <div className="text-sm text-[#525252]">{r.period}</div>
                  <div className="mt-3 text-[16px] text-[#262626]">{r.title}</div>
                  <div className="mt-3 text-sm text-[#525252]">{r.subtitle}</div>
                </div>
              ))}
              {!recent.length && (
                <div className="col-span-full text-sm text-neutral-500">아직 최근 검색이 없습니다.</div>
              )}
            </div>
          </section>
        </div>
      </main>

      {/* footer */}
      <footer className="absolute left-0 bottom-0 w-full h-[125px] bg-white flex items-center border-t border-neutral-200">
        <div className="w-full max-w-[1280px] mx-auto">
          <div className="text-center text-[#525252]">여떠잼</div>
          <div className="text-center text-sm text-[#737373] mt-2">© 2025 여떠잼. All rights reserved.</div>
        </div>
      </footer>
    </div>
  );
}
