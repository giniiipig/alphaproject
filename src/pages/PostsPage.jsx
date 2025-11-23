import { useState } from "react";
import PostCreateForm from "../components/PostCreateForm";
import PostList from "../components/PostList";

export default function PostsPage() {
  const [key, setKey] = useState(0); // 간단 리프레시 키
  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold">Posts</h1>
      <PostCreateForm onCreated={() => setKey((k) => k + 1)} />
      <div key={key}>
        <PostList />
      </div>
    </div>
  );
}
