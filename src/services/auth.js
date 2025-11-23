// 최소 스텁
export function isLoggedIn() {
  return Boolean(localStorage.getItem("token"));
}
