import { useState } from "react";
import { createPost } from "../lib/posts";

export default function PostCreateForm({ authorId = 1, onCreated }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await createPost({ title, content, author_id: authorId });
      setTitle("");
      setContent("");
      onCreated?.();
    } catch (e) {
      setErr(
        e?.response?.data ? JSON.stringify(e.response.data) : e?.message ?? "저장 실패"
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-2 p-3 rounded-xl border">
      <input
        className="w-full border rounded-lg p-2"
        placeholder="제목"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        required
      />
      <textarea
        className="w-full border rounded-lg p-2 h-28"
        placeholder="내용"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        required
      />
      {err && <div className="text-sm text-red-600">{err}</div>}
      <button
        type="submit"
        disabled={busy}
        className="px-3 py-2 rounded-lg bg-black text-white disabled:opacity-50"
      >
        {busy ? "저장 중…" : "작성"}
      </button>
    </form>
  );
}
