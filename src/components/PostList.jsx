import { useEffect, useState } from "react";
import { fetchPosts } from "../lib/posts";

export default function PostList() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        setItems(await fetchPosts());
      } catch (e) {
        setErr(e?.message ?? "불러오기 실패");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <div>불러오는 중…</div>;
  if (err) return <div style={{ color: "crimson" }}>{err}</div>;

  return (
    <div className="space-y-3">
      {items.map((p) => (
        <article key={p.id} className="p-3 rounded-xl border">
          <h3 className="font-semibold">{p.title}</h3>
          <p className="text-sm text-gray-700 whitespace-pre-wrap">{p.content}</p>
          <div className="text-xs text-gray-500 mt-1">
            by {String(p.author ?? p.author_id)}{" "}
            {p.created_at ? `· ${new Date(p.created_at).toLocaleString()}` : ""}
          </div>
        </article>
      ))}
    </div>
  );
}
