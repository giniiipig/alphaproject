import { getApiBase } from "../../lib/env";

// 서버가 제공하는 "시작" 엔드포인트로 이동 (문서: GET /auth/kakao/login/)
export function startKakaoOnServer() {
  const base = getApiBase();
  const origin =
    !base || base === "/"
      ? ""
      : base.endsWith("/api")
      ? base.replace(/\/api$/, "")
      : base;
  window.location.assign(`${origin}/auth/kakao/login/`);
}
