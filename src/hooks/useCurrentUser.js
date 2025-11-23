// src/hooks/useCurrentUser.js
import { useEffect, useState, useCallback } from "react";
import { buildApiUrl } from "../lib/env";

const AUTH_ME_URL = buildApiUrl("/auth/me/");

// 여러 후보 중 첫 truthy 값
const pick = (...args) => args.find((v) => v !== undefined && v !== null && v !== "");

/** 서버에서 온 유저 객체 → 화면에서 쓰기 좋은 형태로 변환 */
function normalizeUser(raw) {
  if (!raw || typeof raw !== "object") return null;

  const name = pick(
    raw.nickname,
    raw.name,
    raw.displayName,
    raw.username,
    raw.email
  );

  const avatar = pick(
    raw.profile_image,
    raw.profileImage,
    raw.avatar,
    raw.image
  );

  return {
    id: raw.id ?? raw.pk ?? null,
    name: name || "손님",
    email: raw.email || null,
    avatar: avatar || null,
    raw,
  };
}

/** URL 쿼리(auth, ok)를 보고 fakeUser 를 초기화 (있으면) */
function initFakeUserFromUrl() {
  if (typeof window === "undefined") return null;

  try {
    const url = new URL(window.location.href);
    const params = url.searchParams;
    const auth = params.get("auth");
    const ok = params.get("ok");

    // 예: /trips?auth=kakao&ok=1 처럼 리다이렉트 되는 경우
    if (ok === "1" && auth) {
      const fake = {
        id: `${auth}-user`,
        name: auth === "kakao" ? "카카오 사용자" : "소셜 사용자",
        email: null,
        avatar: null,
        raw: { provider: auth },
      };
      localStorage.setItem("fakeUser", JSON.stringify(fake));

      // 쿼리 깔끔하게 제거
      params.delete("auth");
      params.delete("ok");
      window.history.replaceState({}, "", url.toString());

      return fake;
    }
  } catch (_) {
    // URL 파싱 실패해도 무시
  }
  return null;
}

/** localStorage 에 저장된 fakeUser 읽기 */
function readFakeUser() {
  try {
    const raw = localStorage.getItem("fakeUser");
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export default function useCurrentUser() {
  // 1) 최초에는 URL → localStorage 순서로 fakeUser 시도
  const [user, setUser] = useState(() => {
    return initFakeUserFromUrl() || readFakeUser();
  });
  const [ready, setReady] = useState(false);

  const fetchMe = useCallback(async () => {
    try {
      // ✅ 프록시를 타도록 /api/auth/me/ 로 호출
      const res = await fetch(AUTH_ME_URL, {
        credentials: "include",
      });

      if (!res.ok) {
        if (res.status === 401 || res.status === 403 || res.status === 404) {
          // 서버 기준으론 비로그인이어도
          // 이미 fakeUser 가 있으면 그대로 두고, 없으면 null
          if (!readFakeUser()) setUser(null);
          return;
        }
        console.error("GET /auth/me/ 실패:", res.status);
        return;
      }

      const data = await res.json();
      const normalized = normalizeUser(data);
      setUser(normalized);
      // 서버에서 진짜 유저를 받았으면 fakeUser 도 업데이트(선택적)
      if (normalized) {
        localStorage.setItem("fakeUser", JSON.stringify(normalized));
      }
    } catch (err) {
      console.error("GET /auth/me/ 에러:", err);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      await fetchMe();
      if (alive) setReady(true);
    })();
    return () => {
      alive = false;
    };
  }, [fetchMe]);

  // 외부에서 로그인/로그아웃 뒤에 다시 호출할 수 있게
  const refresh = useCallback(async () => {
    await fetchMe();
  }, [fetchMe]);

  // Header 에서 쓰는 loading 플래그도 제공 (기존 ready 유지)
  const loading = !ready;

  return { user, ready, loading, refresh };
}
