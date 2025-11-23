// src/components/Header.jsx
import { Link, useNavigate } from "react-router-dom";
import useCurrentUser from "../hooks/useCurrentUser";

function Initials({ name = "U" }) {
  const initials = String(name).trim().slice(0, 2) || "U";
  return (
    <div className="w-8 h-8 rounded-full bg-neutral-800 text-white text-xs flex items-center justify-center">
      {initials}
    </div>
  );
}

export default function Header() {
  const navigate = useNavigate();
  // ✅ logout 위해 refresh도 같이 가져오기
  const { user, loading } = useCurrentUser();

  const handleLoginClick = () => {
    navigate("/login");
  };

  // ✅ 프론트 전용 로그아웃
  const handleLogoutClick = () => {
    try {
      // 우리가 로그인 때 쓰는 정보들 정리
      localStorage.removeItem("fakeUser");
      // 필요하면 accessToken 등 다른 것도 같이 지워도 됨
      // localStorage.removeItem("accessToken");
      // localStorage.removeItem("refreshToken");
    } catch (e) {
      console.error("logout error", e);
    }
    // 새로고침해서 헤더/페이지 상태 초기화
    window.location.href = "/";
  };

  return (
    <header className="w-full h-[73px] bg-white border-b border-neutral-200">
      <div className="max-w-[1280px] h-full mx-auto px-5 flex items-center justify-between">
        <button onClick={() => navigate("/")} className="flex items-center">
          <span className="text-[20px] md:text-[22px] font-medium text-neutral-800">
            여떠잼
          </span>
        </button>

        <nav className="flex items-center gap-6">
          <Link to="/" className="h-10 px-3 text-neutral-600 hover:text-neutral-800">
            홈
          </Link>
          <Link to="/trips" className="h-10 px-3 text-neutral-600 hover:text-neutral-800">
            내 여행
          </Link>

          <div className="h-8 px-2 flex items-center gap-2 rounded">
            {loading ? (
              <>
                <div className="w-8 h-8 rounded-full bg-neutral-200 animate-pulse" />
                <div className="w-16 h-4 bg-neutral-200 rounded animate-pulse" />
              </>
            ) : user ? (
              <>
                {user.avatar ? (
                  <img
                    src={user.avatar}
                    alt={user.name}
                    className="w-8 h-8 rounded-full object-cover border border-neutral-200"
                    onError={(e) => (e.currentTarget.style.display = "none")}
                  />
                ) : (
                  <Initials name={user.name} />
                )}
                <span className="text-sm text-neutral-800 max-w-[160px] truncate">
                  {user.name}
                </span>
                {/* ✅ 로그아웃 버튼 */}
                <button
                  onClick={handleLogoutClick}
                  className="ml-2 h-8 px-3 rounded border border-neutral-300 text-xs text-neutral-700 hover:bg-neutral-50"
                >
                  로그아웃
                </button>
              </>
            ) : (
              <button
                onClick={handleLoginClick}
                className="h-8 px-3 rounded border border-neutral-300 text-sm text-neutral-700 hover:bg-neutral-50"
              >
                로그인
              </button>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}
