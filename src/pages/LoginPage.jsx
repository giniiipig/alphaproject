import React from "react";
import { Link, useNavigate } from "react-router-dom";
import Header from "../components/Header.jsx";
import { getApiBase } from "../lib/env";

/* === 공용 함수들 === */
const API_ORIGIN = (() => {
  const base = getApiBase();
  if (!base || base === "/") return "";
  return base.endsWith("/api") ? base.replace(/\/api$/, "") : base;
})();

const buildAuthUrl = (path) => `${API_ORIGIN}${path}`;

/* === 컴포넌트 === */
export default function LoginPage() {
  const navigate = useNavigate();

  // Google / Naver
  const onGoogle = () => window.location.assign(buildAuthUrl("/auth/google"));
  const onNaver = () => window.location.assign(buildAuthUrl("/auth/naver"));

    // Kakao (프론트에서 fakeUser 저장 + 백엔드 로그인 시작)
  const onKakao = () => {
    // 1) 프론트 전용 "로그인된 사용자" 정보 저장
    const fake = {
      id: "kakao-user",
      name: "카카오 사용자",
      email: null,
      avatar: null,
      raw: { provider: "kakao" },
    };
    localStorage.setItem("fakeUser", JSON.stringify(fake));

    // 2) 백엔드 카카오 로그인 시작 (기존 동작 유지)
    window.location.href = buildAuthUrl("/auth/kakao/login/");
  };


  return (
    <div className="min-h-screen w-full bg-neutral-50">
      <Header />

      {/* Login Section */}
      <section className="min-h-[calc(100vh-80px)] flex items-center justify-center px-6 py-12">
        <div className="max-w-[1200px] w-full mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            
            {/* 왼쪽: 소개 영역 */}
            <div className="text-center lg:text-left">
              <div className="w-32 h-32 bg-neutral-400 rounded-full mx-auto lg:mx-0 mb-8 flex items-center justify-center">
                <img
                  src="/plane.png"
                  alt="여떠잼"
                  className="w-16 h-16 object-contain"
                  draggable="false"
                />
              </div>
              
              <h1 className="text-4xl md:text-5xl text-neutral-800 mb-6 leading-tight">
                여떠잼과 함께<br />특별한 여행을 시작하세요
              </h1>
              
              <p className="text-xl text-neutral-600 mb-8 max-w-xl mx-auto lg:mx-0">
                소셜 로그인으로 간편하게 시작하고, AI가 추천하는 맞춤형 여행 계획을 경험해보세요
              </p>

              <div className="hidden lg:block space-y-4 text-neutral-600">
                
              </div>
            </div>

            {/* 오른쪽: 로그인 폼 */}
            <div className="bg-white rounded-2xl shadow-lg p-8 md:p-12">
              <div className="text-center mb-10">
                <h2 className="text-3xl text-neutral-800 mb-3">로그인</h2>
                <p className="text-lg text-neutral-600">
                  소셜 계정으로 간편하게 시작하세요
                </p>
              </div>

              <div className="space-y-4">
                {/* Google 로그인 */}
                <button
                  onClick={onGoogle}
                  className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-white border-2 border-neutral-300 text-neutral-800 text-lg rounded-lg hover:bg-neutral-50 hover:border-neutral-400 transition-all"
                >
                  <GoogleG className="w-6 h-6" />
                  <span className="font-medium">Google로 시작하기</span>
                </button>

                {/* Kakao 로그인 */}
                <button
                  onClick={onKakao}
                  className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-[#FEE500] text-neutral-900 text-lg rounded-lg hover:brightness-95 transition-all"
                >
                  <img
                    src="/kakao-logo.png"
                    alt="Kakao"
                    className="w-6 h-6 rounded"
                  />
                  <span className="font-semibold">카카오로 시작하기</span>
                </button>

                {/* Naver 로그인 */}
                <button
                  onClick={onNaver}
                  className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-[#03C75A] text-white text-lg rounded-lg hover:brightness-95 transition-all"
                >
                  <NaverN className="w-6 h-6" />
                  <span className="font-semibold">네이버로 시작하기</span>
                </button>
              </div>

              <div className="mt-8 pt-6 border-t border-neutral-200 text-center text-sm text-neutral-500">
                로그인 시{" "}
                <Link to="/terms" className="text-neutral-700 hover:underline">
                  이용약관
                </Link>
                {" "}및{" "}
                <Link to="/privacy" className="text-neutral-700 hover:underline">
                  개인정보처리방침
                </Link>
                에 동의하게 됩니다.
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}

/* === 아이콘 컴포넌트 === */
function GoogleG({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
    >
      <path
        fill="#FFC107"
        d="M43.6 20.5h-1.9v-.1H24v7.2h11.3A11.9 11.9 0 0 1 24 35a12 12 0 1 1 8.5-20.5l5.1-5A20 20 0 1 0 24 44c11 0 20-9 20-20 0-1.2-.1-2.3-.4-3.5z"
      />
      <path
        fill="#FF3D00"
        d="m6.3 14.7 6.2 4.6A12 12 0 0 1 24 12c3.2 0 6.1 1.2 8.3 3.3l5.1-5A20 20 0 0 0 6.3 14.7z"
      />
      <path
        fill="#4CAF50"
        d="M24 44c5.3 0 10.2-2.1 13.8-5.6l-6.4-5.4A11.9 11.9 0 0 1 24 36a12 12 0 0 1-11.5-8.6l-6.2 4.8A20 20 0 0 0 24 44z"
      />
      <path
        fill="#1976D2"
        d="M43.6 20.5H24v7.2h11.3c-.5 2.3-1.8 4.3-3.6 5.7l6.4 5.4c3.8-3.5 6-8.6 6-14.8 0-1.2-.1-2.3-.4-3.5z"
      />
    </svg>
  );
}

function NaverN({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
    >
      <path
        fill="#fff"
        d="M29.5 32h-5.3L18.5 24v8h-4V16h5.2l5.7 8.1V16h4v16z"
      />
    </svg>
  );
}

/* === Footer 컴포넌트 === */
function Footer() {
  const sections = [
    { title: "서비스", items: ["여행 계획", "여행지 정보", "안전 서비스"] },
    { title: "고객지원", items: ["도움말", "문의하기", "FAQ"] },
  ];

  return (
    <footer className="bg-neutral-900 text-white py-16">
      <div className="max-w-[1440px] mx-auto px-6 grid lg:grid-cols-4 gap-12">
        <div>
          <div className="flex items-center mb-6">
            <span className="text-2xl">🗺 여떠잼</span>
          </div>
          <p className="text-lg text-neutral-400">당신의 완벽한 여행 파트너</p>
        </div>

        {sections.map(({ title, items }) => (
          <div key={title}>
            <h4 className="text-lg mb-6">{title}</h4>
            <ul className="space-y-3 text-neutral-400">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}

        <div>
          <h4 className="text-lg mb-6">팔로우</h4>
          <div className="flex flex-col space-y-2 text-neutral-400">
            <span className="hover:text-white cursor-pointer">Facebook</span>
            <span className="hover:text-white cursor-pointer">Instagram</span>
            <span className="hover:text-white cursor-pointer">Twitter</span>
          </div>
        </div>
      </div>

      <div className="border-t border-neutral-800 mt-12 pt-8 text-center text-lg text-neutral-400">
        © 2025 여떠잼. All rights reserved.
      </div>
    </footer>
  );
}
