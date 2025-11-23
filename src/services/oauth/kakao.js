// src/services/oauth/kakao.js

/**
 * 카카오 로그인 시작 (프론트 → 카카오 OAuth authorize)
 * - 인가 코드(code)를 받으면 백엔드가 세션 생성 처리함.
 */
export function beginKakaoLogin() {
  // ✅ 1) localhost/127.0.0.1 불일치 교정
  // 카카오 콘솔 redirect_uri가 localhost로 등록돼 있을 경우,
  // 브라우저가 127.0.0.1:5173으로 열리면 redirect 불일치로 실패하므로 교정.
  if (window.location.hostname === "127.0.0.1") {
    const u = new URL(window.location.href);
    u.hostname = "localhost";
    window.location.replace(u.toString());
    return; // 교체 후 자동 재로드
  }

  // ✅ 2) REST API 키 가져오기 (백엔드 세션 방식에서는 REST 키 사용)
  const clientId =
    import.meta.env.VITE_KAKAO_REST_API_KEY ||
    import.meta.env.VITE_KAKAO_REST_KEY;

  if (!clientId) {
    console.error("[Kakao] REST API 키가 설정되지 않았습니다 (.env 확인)");
    return;
  }

  // ✅ 3) Redirect URI (카카오 콘솔 등록값과 정확히 일치해야 함)
  // 일반적으로: http://localhost:5173/oauth/kakao/callback
  const redirectUri =
    import.meta.env.VITE_KAKAO_REDIRECT_URI ||
    `${window.location.origin}/oauth/kakao/callback`;

  // ✅ 4) CSRF 방지용 state 생성 (같은 탭 내 유지)
  const state = crypto.randomUUID();
  sessionStorage.setItem("kakao_oauth_state", state);

  // ✅ 5) 카카오 로그인 URL 생성
  const authorizeUrl = new URL("https://kauth.kakao.com/oauth/authorize");
  authorizeUrl.searchParams.set("client_id", clientId);
  authorizeUrl.searchParams.set("redirect_uri", redirectUri);
  authorizeUrl.searchParams.set("response_type", "code");
  authorizeUrl.searchParams.set("state", state);

  // ✅ 6) 로그인 페이지로 이동 (팝업 X)
  window.location.assign(authorizeUrl.toString());
}
