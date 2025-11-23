// src/pages/RouteMap.jsx
// -----------------------------------------------------------------------------
// 여떠잼 | 추천 코스 지도 페이지
// - AI 추천(로컬 스토리지 캐시 → 서버 조회)으로 코스를 불러와 Kakao 지도에 시각화
// - 로그인 시 코스 저장/재요청 가능
// - UI: 좌측 리스트(일자/장소), 우측 지도(마커/경로/컨트롤/범례)
// -----------------------------------------------------------------------------

import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import useKakaoMapsLoader from "../hooks/useKakaoMapsLoader";
import useCurrentUser from "../hooks/useCurrentUser";
import Header from "../components/Header.jsx";
import { buildApiUrl } from "../lib/env";

/* ============================================================================
 * 색상 팔레트
 * - Day 인덱스에 따라 도트/라인 색상 순환
 * ========================================================================== */
const DOT_COLORS = [
  "bg-black",
  "bg-neutral-500",
  "bg-neutral-300",
  "bg-blue-600",
  "bg-emerald-600",
  "bg-amber-500",
  "bg-purple-600",
];
const LINE_COLORS = ["#111111", "#6B7280", "#D1D5DB", "#2563EB", "#059669", "#F59E0B", "#7C3AED"];
const dotClass = (i) => DOT_COLORS[i % DOT_COLORS.length];

/* ============================================================================
 * 응답 정규화 유틸
 * - 서버/AI 포맷 차이를 흡수하여 일관된 { meta, days[] } 구조로 변환
 * ========================================================================== */
/**
 * 서버 응답을 여떠잼 코스 표준형으로 정규화한다.
 * @param {any} data - 서버에서 받은 JSON
 * @returns {{meta:object, days:Array<{date:string, places:Array}>}|null}
 */
function normalizeCourse(data) {
  if (data?.days?.length) return data;

  // 과거 포맷(itinerary)을 days 포맷으로 변환
  if (data?.itinerary?.length) {
    return {
      meta: data.meta || {},
      days: data.itinerary.map((d) => ({
        date: d.date || "",
        places: (d.places || []).map((p) => ({
          name: p.name,
          desc: p.description || p.desc || "",
          time: p.time || "",
          lat: Number(p.lat),
          lng: Number(p.lng),
        })),
      })),
    };
  }
  return null;
}

/* ============================================================================
 * AI 결과 변환
 * - 로컬 캐시에 저장된 AI 추천 Top-K 리스트 → N일 코스로 빌드 (★패치 1)
 * ========================================================================== */
/**
 * AI items 배열을 N일 코스로 빌드한다.
 * @param {Array} aiItems - [{title/coord/lat/lon/...}, ...]
 * @returns {{meta:object, days:Array}|null}
 */
function buildCourseFromAI(aiItems) {
  if (!Array.isArray(aiItems) || aiItems.length === 0) return null;

  // 1) 선호(기간/지역/이동수단/테마) 읽기
  let prefs = null;
  try {
    prefs = JSON.parse(localStorage.getItem("tripPreferences") || "null");
  } catch {}
  const startDate = prefs?.startDate || "";
  const endDate = prefs?.endDate || "";
  const region = prefs?.region || "AI 추천";
  const transport = prefs?.transport || "";
  const themeText = Array.isArray(prefs?.theme)
    ? prefs.theme.join("/")
    : prefs?.theme || "";

  // 2) 일수 계산(기간 없으면 1일)
  const d1 = startDate ? new Date(startDate) : null;
  const d2 = endDate ? new Date(endDate) : null;
  const dayCount =
    d1 && d2 ? Math.max(1, Math.floor((+d2 - +d1) / 86400000) + 1) : 1;

  // 3) 좌표/이름 정규화
  const normalized = aiItems
    .map((it, i) => {
      const lat = Number(
        it.lat ?? it.latitude ?? it.coord?.lat ?? it.coords?.lat
      );
      const lng = Number(
        it.lon ?? it.lng ?? it.longitude ?? it.coord?.lon ?? it.coords?.lon
      );
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
      return {
        name: it.title || it.name || `장소 ${i + 1}`,
        desc: it.overview || it.desc || "",
        time: "",
        lat,
        lng,
      };
    })
    .filter(Boolean);
  if (!normalized.length) return null;

  // 4) 라운드로빈 분배
  const buckets = Array.from({ length: dayCount }, () => []);
  normalized.forEach((p, idx) => buckets[idx % dayCount].push(p));

  // 5) 날짜 라벨
  const days = buckets.map((places, i) => {
    let dateStr = "";
    if (d1) {
      const di = new Date(+d1 + i * 86400000);
      dateStr = di.toISOString().slice(0, 10);
    }
    return { date: dateStr, places };
  });

  return {
    meta: { startDate, endDate, region, transport, theme: themeText },
    days,
  };
}

export default function RouteMap() {
  // ---------------------------------------------------------------------------
  // 외부 훅 / 라우터
  // ---------------------------------------------------------------------------
  const kakaoReady = useKakaoMapsLoader();
  const { user } = useCurrentUser();
  const navigate = useNavigate();

  // ---------------------------------------------------------------------------
  // 상태: 코스/로딩/에러/저장 알림
  // ---------------------------------------------------------------------------
  const [course, setCourse] = useState(null);
  const [loadingCourse, setLoadingCourse] = useState(true);
  const [courseError, setCourseError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");

  // ---------------------------------------------------------------------------
  // Kakao 지도 객체 및 오버레이 레지스트리
  // - mapRef: DOM 컨테이너
  // - mapObj: kakao.maps.Map 인스턴스
  // - overlaysRef: { markers[], polylines[] } (언마운트/업데이트 시 정리)
  // ---------------------------------------------------------------------------
  const mapRef = useRef(null);
  const mapObj = useRef(null);
  const overlaysRef = useRef({ markers: [], polylines: [] });
  const [mapReady, setMapReady] = useState(false);

  // 현재 활성 Day (좌측 Day 버튼으로 변경)
  const [activeDayIdx, setActiveDayIdx] = useState(0);

  /* ==========================================================================
   * ① 코스 불러오기
   * - 우선순위: (1) 로컬 AI 캐시 → (2) 로그인 시 서버 최신 코스 GET → 없으면 POST 생성
   * - 예외: 게스트는 AI 캐시만 사용
   *   (★패치 2: latest_ai_items 우선 + ai_result 하위호환)
   * ======================================================================== */
  useEffect(() => {
    const controller = new AbortController();

    (async () => {
      try {
        setLoadingCourse(true);
        setCourseError(null);

        // (1) 로컬 AI 캐시 (latest_ai_items 우선, 없으면 ai_result 하위호환)
        const aiItems = (() => {
          try {
            const a = JSON.parse(
              localStorage.getItem("latest_ai_items") || "null"
            );
            if (Array.isArray(a) && a.length) return a;
          } catch {}
          try {
            const b = JSON.parse(localStorage.getItem("ai_result") || "[]");
            return Array.isArray(b) ? b : [];
          } catch {
            return [];
          }
        })();
        const aiCourse = buildCourseFromAI(aiItems);
        if (aiCourse) {
          setCourse(aiCourse);
          setActiveDayIdx(0);
          return; // 캐시가 있으면 서버 통신 생략
        }

        // (2) 로그인 사용자: 서버 최신 코스 조회 (없으면 생성)
        if (user) {
          const token =
            localStorage.getItem("accessToken") ||
            localStorage.getItem("token") ||
            "";
          let res = await fetch(buildApiUrl("/recommendations/course/latest"), {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            credentials: "include",
            signal: controller.signal,
          });

          // 404 → 선호정보로 신규 생성
          if (res.status === 404) {
            const prefs = JSON.parse(
              localStorage.getItem("tripPreferences") || "null"
            );
            if (prefs) {
              res = await fetch(buildApiUrl("/recommendations/course"), {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                credentials: "include",
                body: JSON.stringify(prefs),
                signal: controller.signal,
              });
            }
          }

          if (res && res.ok) {
            const json = await res.json();
            const normalized = normalizeCourse(json);
            if (normalized) {
              setCourse(normalized);
              setActiveDayIdx(0);
            }
          }
        }
      } catch (e) {
        setCourseError(e?.message || "코스 불러오기 실패");
        setCourse(null);
      } finally {
        setLoadingCourse(false);
      }
    })();

    return () => controller.abort();
  }, [user]);

  /* ==========================================================================
   * ② 지도 생성/해제
   * - 최초 1회 생성, 언마운트 시 오버레이/맵 정리
   * ======================================================================== */
  useEffect(() => {
    if (!kakaoReady || !mapRef.current || mapObj.current) return;

    const kakao = window.kakao;
    const center = new kakao.maps.LatLng(37.5665, 126.978);
    const map = new kakao.maps.Map(mapRef.current, { center, level: 9 });
    mapObj.current = map;
    setMapReady(true);

    // 정리
    return () => {
      overlaysRef.current.markers.forEach((m) => m.setMap(null));
      overlaysRef.current.polylines.forEach((p) => p.setMap(null));
      overlaysRef.current = { markers: [], polylines: [] };
      mapObj.current = null;
      setMapReady(false);
    };
  }, [kakaoReady]);

  /* ==========================================================================
   * ③ 오버레이(마커/경로) 렌더링
   * - activeDayIdx, course 변경 시 재그리기
   * - 이전 오버레이는 반드시 제거
   * ======================================================================== */
  useEffect(() => {
    if (!kakaoReady || !mapObj.current || !course) return;

    const kakao = window.kakao;
    const map = mapObj.current;

    // 기존 오버레이 정리
    overlaysRef.current.markers.forEach((m) => m.setMap(null));
    overlaysRef.current.polylines.forEach((p) => p.setMap(null));
    overlaysRef.current = { markers: [], polylines: [] };

    const day = course.days[activeDayIdx];
    const points = day?.places || [];
    if (!points.length) return;

    // (1) 마커 + 인포윈도우
    points.forEach((p, idx) => {
      const pos = new kakao.maps.LatLng(p.lat, p.lng);
      const marker = new kakao.maps.Marker({ position: pos, map });
      const iwContent = `
        <div style="padding:8px 10px; font-size:12px;">
          <div style="font-weight:600; margin-bottom:4px;">${idx + 1}. ${p.name}</div>
          <div style="color:#666">${p.desc || ""}</div>
          <div style="color:#888; margin-top:2px;">${p.time || ""}</div>
        </div>`;
      const infowindow = new kakao.maps.InfoWindow({ content: iwContent });
      kakao.maps.event.addListener(marker, "click", () => infowindow.open(map, marker));
      overlaysRef.current.markers.push(marker);
    });

    // (2) 경로(Polyline)
    const path = points.map((p) => new kakao.maps.LatLng(p.lat, p.lng));
    const polyline = new kakao.maps.Polyline({
      map,
      path,
      strokeWeight: 5,
      strokeColor: LINE_COLORS[activeDayIdx % LINE_COLORS.length],
      strokeOpacity: 0.9,
      strokeStyle: "solid",
    });
    overlaysRef.current.polylines.push(polyline);

    // (3) 보기 범위 자동 맞춤
    const bounds = new kakao.maps.LatLngBounds();
    path.forEach((ll) => bounds.extend(ll));
    map.setBounds(bounds, 40, 40, 40, 40);
  }, [activeDayIdx, kakaoReady, course]);

  /* ==========================================================================
   * ④ 지도 컨트롤(확대/축소/내 위치)
   * ======================================================================== */
  const handleZoom = (dir) => {
    if (!mapObj.current) return;
    mapObj.current.setLevel(
      mapObj.current.getLevel() + (dir === "out" ? 1 : -1)
    );
  };

  const handleLocate = () => {
    if (!mapObj.current || !navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition((pos) => {
      const { latitude, longitude } = pos.coords;
      const latlng = new window.kakao.maps.LatLng(latitude, longitude);
      mapObj.current.setCenter(latlng);
    });
  };

  /* ==========================================================================
   * 서버 저장 페이로드 변환
   * - 프론트 표준 코스 → 서버 수신 구조로 변환
   * ======================================================================== */
  function toBackendCoursePayload(c) {
    if (!c?.days?.length) return null;
    return {
      meta: {
        region: c.meta?.region || "",
        startDate: c.meta?.startDate || "",
        endDate: c.meta?.endDate || "",
        transport: c.meta?.transport || "",
        theme: c.meta?.theme || "",
      },
      days: c.days.map((d) => ({
        date: d.date || "",
        places: (d.places || []).map((p) => ({
          name: p.name,
          desc: p.desc || "",
          time: p.time || "",
          lat: Number(p.lat),
          lng: Number(p.lng),
        })),
      })),
    };
  }

  /* ==========================================================================
   * 저장(로그인 필요) → (★패치 3) 비로그인 시에도 로컬 "게스트 보관함" 저장
   * - 실패 시 메시지 노출, 성공 시 /mypage 이동
   * ======================================================================== */
  const handleSave = async () => {
    if (!course) return;

    // 게스트 저장
    if (!user) {
      try {
        const card = {
          id: Date.now(),
          title: course.meta?.region
            ? `${course.meta.region} ${course.days?.length || 1}일`
            : `여행 ${course.days?.length || 1}일`,
          start_date: course.meta?.startDate || "",
          end_date: course.meta?.endDate || "",
          cover_url: "",
          places_summary: (course?.days?.[0]?.places || [])
            .slice(0, 3)
            .map((p) => p.name)
            .join(" · "),
          course, // 전체 코스 원본
          created_at: new Date().toISOString(),
        };
        const arr = JSON.parse(localStorage.getItem("guest_trips") || "[]");
        arr.unshift(card);
        localStorage.setItem("guest_trips", JSON.stringify(arr.slice(0, 50)));
        setSaveMsg("게스트 보관함에 저장했어요! (마이페이지에서 확인)");
        setTimeout(() => setSaveMsg(""), 2800);
      } catch (e) {
        setSaveMsg("로컬 저장 실패");
        setTimeout(() => setSaveMsg(""), 2800);
      }
      return;
    }

    // 로그인 사용자 서버 저장
    const payload = toBackendCoursePayload(course);
    if (!payload) {
      setSaveMsg("저장할 코스가 없습니다.");
      return;
    }

    setSaving(true);
    setSaveMsg("");
    try {
      const token =
        localStorage.getItem("accessToken") ||
        localStorage.getItem("token") ||
        "";
      const common = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        credentials: "include",
        body: JSON.stringify(payload),
      };

      let res = await fetch(buildApiUrl("/recommendations/course/save"), common);
      if (res.status === 404)
        res = await fetch(buildApiUrl("/recommendations/course"), common);
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || `HTTP ${res.status}`);
      }
      navigate("/mypage", { replace: false });
    } catch (e) {
      setSaveMsg(`저장 실패: ${e?.message || e}`);
      setTimeout(() => setSaveMsg(""), 4000);
    } finally {
      setSaving(false);
    }
  };

  /* ==========================================================================
   * 경로 재요청
   * - 게스트: 플랜 페이지로 유도
   * - 로그인: 서버에 생성 요청 후 최신 코스 세팅
   * ======================================================================== */
  const refetchCourse = async () => {
    if (!user) {
      navigate("/plan");
      return;
    }
    try {
      setLoadingCourse(true);
      setCourseError(null);

      const token =
        localStorage.getItem("accessToken") ||
        localStorage.getItem("token") ||
        "";
      const prefs =
        JSON.parse(localStorage.getItem("tripPreferences") || "null") || {};

      const res = await fetch(buildApiUrl("/recommendations/course"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        credentials: "include",
        body: JSON.stringify(prefs),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const json = await res.json();
      const normalized = normalizeCourse(json) || null;
      setCourse(normalized);
      setActiveDayIdx(0);
    } catch (e) {
      setCourseError(e?.message || "추천 재요청 실패");
      setCourse(null);
    } finally {
      setLoadingCourse(false);
    }
  };

  // 파생 값 (렌더 편의)
  const meta = course?.meta || {};
  const dayCount = course?.days?.length || 0;

  /* ==========================================================================
   * 렌더
   * ======================================================================== */
  return (
    <div className="min-h-screen w-full bg-[#FAFAFA]">
      <Header />

      {/* 상단 조건 바 */}
      <div className="w-full h-[63px] bg-[#F5F5F5] border-b border-neutral-200">
        <div className="max-w-[1440px] mx-auto h-full px-6 flex items-center justify-between">
          <div className="flex items-center gap-6 text-sm text-neutral-700">
            <span>
              {meta.startDate ? `${meta.startDate} - ${meta.endDate}` : "여행일정 미정"}
              {dayCount ? ` (${dayCount}일)` : ""}
            </span>
            <span>{meta.region || "지역 미정"}</span>
            <span>{meta.transport || "이동수단 미정"}</span>
            <span>{meta.theme || "테마 미정"}</span>
          </div>
          <button
            onClick={refetchCourse}
            className="h-[30px] px-3 text-xs border border-neutral-300 rounded hover:bg-neutral-50"
          >
            경로 추천 다시 받기
          </button>
        </div>
      </div>

      {/* 본문 그리드 */}
      <main className="max-w-[1440px] mx-auto px-6 py-0">
        <div className="grid grid-cols-1 md:grid-cols-[426.66px_1fr] gap-0">
          {/* 좌측 패널: Day/장소 리스트 */}
          <aside className="bg-white h-[600px] border-r border-neutral-200">
            {/* 패널 헤더 */}
            <div className="h-[105px] border-b border-neutral-200 px-6 py-4">
              <div className="text-[20px] leading-[28px] text-neutral-800 mb-1">추천 여행 코스</div>
              <div className="text-sm text-neutral-600">
                {loadingCourse
                  ? "코스를 불러오는 중…"
                  : courseError
                  ? "코스를 불러오지 못했습니다"
                  : dayCount
                  ? `${dayCount}일 일정`
                  : "표시할 코스가 없습니다"}
              </div>
            </div>

            {/* Day 선택 */}
            {!!dayCount && (
              <div className="p-4 flex gap-2">
                {(course?.days || []).map((_, idx) => (
                  <button
                    key={idx}
                    onClick={() => setActiveDayIdx(idx)}
                    className={`px-3 h-8 rounded border text-sm ${
                      activeDayIdx === idx
                        ? "bg-black text-white border-black"
                        : "border-neutral-300 text-neutral-700 hover:bg-neutral-50"
                    }`}
                  >
                    Day {idx + 1}
                  </button>
                ))}
              </div>
            )}

            {/* 장소 리스트 / 빈 상태 */}
            <div className="px-4 space-y-3 overflow-auto" style={{ height: 600 - 105 - 48 - 16 }}>
              {(course?.days?.[activeDayIdx]?.places || []).map((p, i) => (
                <div key={p.name + i} className="border border-neutral-200 rounded-lg bg-white p-3">
                  <div className="flex gap-3">
                    <div className="w-12 h-12 bg-neutral-200 rounded" />
                    <div className="flex-1">
                      <div className="text-sm text-neutral-900">{p.name}</div>
                      <div className="text-xs text-neutral-600 mt-0.5">{p.desc}</div>
                      <div className="text-xs text-neutral-500 mt-1">{p.time}</div>
                    </div>
                  </div>
                </div>
              ))}

              {!loadingCourse && !dayCount && (
                <div className="text-sm text-neutral-500 p-4">
                  표시할 경로가 없습니다.{" "}
                  {user ? (
                    "상단의 ‘경로 추천 다시 받기’를 눌러 생성하세요."
                  ) : (
                    <button onClick={() => navigate("/plan")} className="underline">
                      AI 추천 받으러 가기
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* 액션 버튼 / 상태 메시지 */}
            <div className="p-4 flex flex-col gap-2">
              <div className="flex gap-3">
                <button
                  onClick={handleSave}
                  disabled={saving || !course}
                  className="flex-1 h-[56px] bg-neutral-900 text-white rounded-lg disabled:opacity-60"
                >
                  {saving ? "저장 중…" : "일정 저장"}
                </button>
                <button
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(window.location.href);
                      setSaveMsg("링크가 클립보드에 복사되었습니다.");
                      setTimeout(() => setSaveMsg(""), 2500);
                    } catch {
                      setSaveMsg("복사 실패. 주소창의 URL을 직접 복사해주세요.");
                      setTimeout(() => setSaveMsg(""), 2500);
                    }
                  }}
                  className="flex-1 h-[56px] border border-neutral-300 rounded-lg text-neutral-700 hover:bg-neutral-50"
                >
                  공유하기
                </button>
              </div>
              {saveMsg && <div className="text-xs text-neutral-700 px-1">{saveMsg}</div>}
              {!user && (
                <div className="text-xs text-neutral-500 px-1">
                  게스트 모드입니다. 이 일정은 로그인 없이도 ‘마이페이지 &gt; 게스트 보관함’에서 볼 수 있어요.
                </div>
              )}
            </div>
          </aside>

          {/* 우측 지도 */}
          <section className="relative h-[600px] bg-neutral-200">
            <div ref={mapRef} className="absolute inset-0" />

            {/* 지도 컨트롤(확대/축소/내 위치) */}
            <div className="absolute right-4 top-4 flex flex-col gap-2 z-10">
              <button onClick={() => handleZoom("in")} className="w-10 h-10 bg-white border border-neutral-300 rounded-lg">
                ＋
              </button>
              <button onClick={() => handleZoom("out")} className="w-10 h-10 bg-white border border-neutral-300 rounded-lg">
                －
              </button>
              <button onClick={handleLocate} className="w-10 h-10 bg-white border border-neutral-300 rounded-lg">
                ⌖
              </button>
            </div>

            {/* 범례 */}
            {!!dayCount && (
              <div className="absolute left-4 bottom-4 bg-white border border-neutral-300 rounded-lg px-3 py-2 text-xs text-neutral-700 flex items-center gap-4">
                {course.days.map((_, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <span className={`inline-block w-3 h-3 rounded-full ${dotClass(idx)}`} />
                    Day {idx + 1}
                  </div>
                ))}
              </div>
            )}

            {/* 로딩 오버레이 */}
            {(!kakaoReady || !mapReady) && (
              <div className="absolute inset-0 flex items-center justify-center text-neutral-600">
                지도를 불러오는 중…
              </div>
            )}
          </section>
        </div>
      </main>

      {/* 푸터 */}
      <footer className="h-[125px] bg-white border-t border-neutral-200 flex items-center">
        <div className="max-w-[1440px] mx-auto w-full px-6 text-center">
          <div className="text-neutral-600">여떠잼</div>
          <div className="text-sm text-neutral-500 mt-2">© 2025 여떠잼. All rights reserved.</div>
        </div>
      </footer>
    </div>
  );
}
