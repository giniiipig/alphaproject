// src/hooks/useKakaoMapsLoader.js
import { useEffect, useState } from "react";

export default function useKakaoMapsLoader() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null); // (옵션) 에러 확인용

  useEffect(() => {
    const KEY =
      import.meta.env.VITE_KAKAO_MAP_KEY ||        // ✅ alias 1
      import.meta.env.VITE_KAKAO_JAVASCRIPT_KEY || // 기존
      import.meta.env.VITE_KAKAO_JS_KEY;           // 기존

    if (!KEY) {
      const msg = "[Kakao] JS key missing. Set VITE_KAKAO_MAP_KEY (or VITE_KAKAO_JAVASCRIPT_KEY) and restart dev server.";
      console.error(msg);
      setError(msg);
      return;
    }

    const onSdkLoaded = () => {
      const k = window.kakao;
      if (k?.maps?.load) {
        k.maps.load(() => setReady(true));
      } else if (k?.maps) {
        setReady(true);
      } else {
        const msg = "[Kakao] window.kakao.maps not available after load.";
        console.error(msg);
        setError(msg);
      }
    };

    // 이미 준비됨
    if (window.kakao?.maps) {
      onSdkLoaded();
      return;
    }

    // 기존 스크립트 재사용/검증
    const id = "kakao-maps-sdk";
    let script = document.getElementById(id);

    // 잘못된 키로 붙어 있으면 교체
    if (script && !script.src.includes(KEY)) {
      script.remove();
      script = null;
    }

    if (!script) {
      script = document.createElement("script");
      script.id = id;
      script.setAttribute("data-kakao", "maps");
      script.async = true;
      script.defer = true; // ✅ 렌더 블로킹 방지
      script.src = `https://dapi.kakao.com/v2/maps/sdk.js?autoload=false&appkey=${KEY}&libraries=services`;
      script.addEventListener("load", onSdkLoaded);
      script.addEventListener("error", (e) => {
        const msg = "[Kakao] SDK script load failed (check network/domain/key).";
        console.error(msg, e);
        setError(msg);
      });
      document.head.appendChild(script);
    } else {
      // 이미 DOM에 있으면 load 콜백만 보장
      script.addEventListener("load", onSdkLoaded);
      if (document.readyState === "complete" && window.kakao?.maps) {
        onSdkLoaded();
      }
    }

    return () => {
      script?.removeEventListener?.("load", onSdkLoaded);
    };
  }, []);

  // 필요 없으면 ready만 반환해도 OK
  return ready; // 또는 { ready, error }
}
