// src/pages/MyPage.jsx
import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Header from "../components/Header.jsx";

function TripCard({ trip, onOpen, onDelete }) {
  const dates =
    trip.start_date && trip.end_date
      ? `${trip.start_date} - ${trip.end_date}`
      : "";

  return (
    <div
      className="rounded-xl border border-neutral-200 overflow-hidden bg-white cursor-pointer hover:shadow-sm transition-shadow"
      onClick={() => onOpen(trip)}
    >
      <div className="h-44 bg-neutral-200 flex items-center justify-center text-neutral-600">
        {trip.cover_url ? (
          <img
            src={trip.cover_url}
            alt={trip.title}
            className="w-full h-full object-cover"
          />
        ) : (
          <span className="px-4">
            {trip.title || "저장된 여행 이미지"}
          </span>
        )}
      </div>
      <div className="p-5">
        <div className="font-semibold text-neutral-900">
          {trip.title || "저장된 여행"}
        </div>
        {dates && (
          <div className="text-sm text-neutral-600 mt-1">{dates}</div>
        )}
        {trip.places_summary && (
          <div className="text-sm text-neutral-500 mt-2 line-clamp-1">
            {trip.places_summary}
          </div>
        )}

        <div
          className="mt-4 text-xs text-neutral-500"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(trip);
          }}
        >
          삭제
        </div>
      </div>
    </div>
  );
}

export default function MyPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [guestTrips, setGuestTrips] = useState([]);

  // ✅ (선택) /trips?auth=kakao&ok=1 같은 리다이렉트 처리 → fakeUser 저장
  useEffect(() => {
    try {
      const params = new URLSearchParams(location.search);
      const auth = params.get("auth");
      const ok = params.get("ok");

      if (auth && ok === "1") {
        const fake = {
          id: `${auth}-user`,
          name: auth === "kakao" ? "카카오 사용자" : "소셜 사용자",
          email: null,
          avatar: null,
        };
        localStorage.setItem("fakeUser", JSON.stringify(fake));

        // URL에서 쿼리 정리
        params.delete("auth");
        params.delete("ok");
        const base = `${location.pathname}${params.toString() ? "?" + params.toString() : ""}`;
        window.history.replaceState({}, "", base);
      }
    } catch {
      // 실패해도 그냥 무시
    }
  }, [location.pathname, location.search]);

  // 게스트 보관함 로드 (기존 기능)
  useEffect(() => {
    try {
      const arr = JSON.parse(localStorage.getItem("guest_trips") || "[]");
      setGuestTrips(Array.isArray(arr) ? arr : []);
    } catch {
      setGuestTrips([]);
    }
  }, []);

  const handleOpenTrip = (trip) => {
    if (!trip.course) {
      alert("이 일정에 저장된 코스 정보가 없습니다.");
      return;
    }
    // 선택된 코스를 RouteMap에서 최우선으로 사용하도록 저장
    localStorage.setItem("selected_course", JSON.stringify(trip.course));
    navigate("/map");
  };

  const handleDeleteTrip = (trip) => {
    const next = guestTrips.filter(
      (t) => (t.id || t.created_at) !== (trip.id || trip.created_at)
    );
    setGuestTrips(next);
    localStorage.setItem("guest_trips", JSON.stringify(next));
  };

  const handleClearAll = () => {
    if (!window.confirm("저장된 모든 일정을 삭제할까요?")) return;
    setGuestTrips([]);
    localStorage.setItem("guest_trips", "[]");
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      <Header />

      {/* 상단 바 */}
      <div className="w-full h-[63px] bg-[#F5F5F5] border-b border-neutral-200">
        <div className="max-w-[1440px] mx-auto h-full px-6 flex items-center justify-between">
          <div className="text-neutral-800 font-medium">내 여행 일정</div>
          <button
            onClick={() => navigate("/plan")}
            className="h-[30px] px-3 text-xs border border-neutral-300 rounded hover:bg-neutral-50"
          >
            새 일정 만들기
          </button>
        </div>
      </div>

      {/* 본문 */}
      <main className="max-w-[1440px] mx-auto px-6 py-8">
        <div className="mb-4 flex items-center justify-between">
          <div className="text-sm text-neutral-700">
          </div>
          {guestTrips.length > 0 && (
            <button
              onClick={handleClearAll}
              className="px-3 h-9 rounded-lg border border-neutral-300 text-neutral-700 hover:bg-neutral-50 text-sm"
            >
              모두 삭제
            </button>
          )}
        </div>

        {guestTrips.length === 0 ? (
          <div className="h-40 flex items-center justify-center text-neutral-600 border rounded-xl bg-white">
            저장된 일정이 없습니다. 지도 페이지에서 일정을 저장해 보세요.
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {guestTrips.map((t) => (
              <TripCard
                key={t.id || t.created_at}
                trip={t}
                onOpen={handleOpenTrip}
                onDelete={handleDeleteTrip}
              />
            ))}
          </div>
        )}
      </main>

      {/* 푸터 */}
      <footer className="h-[125px] bg-white border-t border-neutral-200 flex items-center">
        <div className="max-w-[1440px] mx-auto w-full px-6 text-center">
          <div className="text-neutral-600">여떠잼</div>
          <div className="text-sm text-neutral-500 mt-2">
            © 2025 여떠잼. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
