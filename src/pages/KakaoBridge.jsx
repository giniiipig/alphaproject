import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { buildApiUrl } from "../lib/env";

export default function KakaoBridge() {
  const nav = useNavigate();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const ticket = params.get("ticket");
    if (!ticket) { nav("/trips?auth=kakao&ok=0&err=no_ticket"); return; }

    // same-origin 인식(/api) + credentials 포함
    fetch(buildApiUrl("/auth/finalize/"), {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ ticket }).toString(),
      credentials: "include",
    })
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(() => nav("/trips?auth=kakao&ok=1", { replace: true }))
    .catch(() => nav("/trips?auth=kakao&ok=0&err=finalize_fail", { replace: true }));
  }, [nav]);

  return <div style={{padding:16}}>카카오 로그인 마무리 중...</div>;
}
