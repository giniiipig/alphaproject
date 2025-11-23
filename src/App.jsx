// src/App.jsx
import React from "react";
import { Routes, Route } from "react-router-dom";

import Home from "./pages/Home.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RouteMap from "./pages/RouteMap.jsx";
import PlanTrip from "./pages/PlanTrip.jsx";
import MyPage from "./pages/MyPage.jsx";
import WriteReview from "./pages/WriteReview.jsx";
import KakaoBridge from "./pages/KakaoBridge";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/plan" element={<PlanTrip />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/map" element={<RouteMap />} />
      <Route path="/mypage" element={<MyPage />} />
      <Route path="/trips" element={<MyPage />} />
      <Route path="/kakao-bridge" element={<KakaoBridge/>} />

      {/* ❌ 프론트 콜백 제거 — 백엔드가 콜백을 처리함 */}
      {/* <Route path="/auth/kakao/callback" element={<KakaoCallback />} /> */}

      <Route path="/reviews/new" element={<WriteReview />} />
      <Route path="*" element={<Home />} />
    </Routes>
  );
}

