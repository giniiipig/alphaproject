// src/lib/posts.js
import api from "./api";

export async function fetchPosts() {
  const { data } = await api.get("/posts/");
  return data;
}

export async function createPost({ title, content, author_id }) {
  const { data } = await api.post(
    "/posts/create/",
    { title, content, author_id },
    { headers: { "Content-Type": "application/json" } }
  );
  return data;
}
