// src/pages/oauth/KakaoCallback.jsx
import React, { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import axios from "axios";
import { getApiBase } from "../../lib/env";

export default function KakaoCallback() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState("");
  const onceRef = useRef(false);

  const API_BASE = getApiBase();
  const ORIGIN_BASE =
    API_BASE === "/"
      ? "/"
      : API_BASE.endsWith("/api")
      ? API_BASE.replace(/\/api$/, "") || "/"
      : API_BASE;
  const defaultAuthBase = ORIGIN_BASE === "/" ? "/auth" : `${ORIGIN_BASE}/auth`;

  // ✅ AUTH 전용 베이스 (기본: API origin + /auth)
  const AUTH_BASE = (import.meta.env.VITE_AUTH_BASE || defaultAuthBase).replace(/\/$/, "");

  // ✅ 세션 쿠키 전달을 위해 루트(또는 API origin) 기준으로 생성
  const api = axios.create({ baseURL: ORIGIN_BASE, withCredentials: true });

  // CSRF 쿠키 있으면 헤더로 첨부 (Django 기본 쿠키명: csrftoken)
  api.interceptors.request.use((config) => {
    const m = document.cookie.match(/(^|;\s*)csrftoken=([^;]+)/);
    if (m) config.headers["X-CSRFToken"] = decodeURIComponent(m[2]);
    return config;
  });

  useEffect(() => {
    if (onceRef.current) return;
    onceRef.current = true;

    const code  = params.get("code");
    const state = params.get("state");
    const saved = sessionStorage.getItem("kakao_oauth_state");

    if (!code)  { setError("인가 코드가 없습니다."); return; }
    if (!state || state !== saved) { setError("상태값 검증 실패"); return; }

    (async () => {
      try {
        // 🔸 Kakao 콘솔에 등록한 값과 '완전히 동일'해야 함 (프로토콜/호스트/포트/슬래시)
        const redirect_uri =
          import.meta.env.VITE_KAKAO_REDIRECT_URI ||
          `${window.location.origin}/oauth/kakao/callback`;

        // 프론트는 code만 서버로 전달 (카카오 토큰 교환은 백엔드)
        await api.post(`${AUTH_BASE}/kakao/callback/`, { code, redirect_uri, state });

        sessionStorage.removeItem("kakao_oauth_state");
        navigate("/trips", { replace: true });
      } catch (e) {
        console.error(e);
        const msg =
          e?.response?.data?.detail ||
          e?.response?.data?.message ||
          e?.message ||
          "로그인 처리 중 오류";
        setError(String(msg));
      }
    })();
  }, []);

  if (error) return <div className="p-6 text-red-500">{error}</div>;
  return <div className="p-6">카카오 로그인 처리 중…</div>;
}
