import { api } from "@/lib/api";

export async function createComment({ post_id, content, author_id }) {
  const { data } = await api.post("/comments/create/", { post_id, content, author_id });
  return data;  // { message, comment_id, ... }
}
