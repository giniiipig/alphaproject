import { api } from "@/lib/api";

export async function getPosts() {
  const { data } = await api.get("/posts/");              // -> /api/posts/
  return data;  // { posts: [...], total_count: n }
}

export async function createPost({ title, content, author_id }) {
  const { data } = await api.post("/posts/create/", { title, content, author_id });
  return data;  // { message, post_id, ... }
}
