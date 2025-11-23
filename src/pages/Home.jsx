import React from "react";
import { useNavigate } from "react-router-dom";
import Testimonials from "../components/Testimonials.jsx";
import Header from "../components/Header.jsx";

export default function Home() {
  const navigate = useNavigate();

  const features = [
    {
      img: `${import.meta.env.BASE_URL}distance.png`,
      alt: "맞춤형 여행 계획",
      title: "맞춤형 여행 계획",
      desc: "AI가 분석한 당신의 취향에 맞는 완벽한 여행 루트를 제안합니다",
    },
    {
      img: `${import.meta.env.BASE_URL}audience.png`,
      alt: "여행지 정보",
      title: "여행지 정보",
      desc: "여행지와 관련된 정보를 한눈에 제공합니다",
    },
    {
      img: `${import.meta.env.BASE_URL}shield.png`,
      alt: "안전한 여행",
      title: "안전한 여행",
      desc: "실시간 안전 정보와 비상 연락망으로 걱정 없는 여행을 보장합니다",
    },
  ];

  return (
    <div className="min-h-screen w-full bg-neutral-50">
      <Header />

      {/* Hero */}
      <section className="h-[800px] flex flex-col items-center justify-center px-6">
        <div className="max-w-[1200px] text-center mx-auto">
          <div className="w-32 h-32 bg-neutral-400 rounded-full mx-auto mb-8 flex items-center justify-center">
            <img
              src="/plane.png"
              alt="여떠잼"
              className="w-16 h-16 object-contain"
              draggable="false"
            />
          </div>

          <h1 className="text-4xl md:text-6xl text-neutral-800 mb-6 leading-tight">
            당신만의 특별한 여행을<br className="hidden md:block" />시작하세요
          </h1>
          <p className="text-xl md:text-2xl text-neutral-600 mb-12 max-w-4xl mx-auto">
            맞춤형 여행 계획부터 여행지 정보까지, 모든 것을 한 곳에서
          </p>

          <button
            onClick={() => navigate("/plan")}
            className="px-8 py-3 bg-neutral-800 text-white text-lg rounded-lg hover:bg-neutral-700 transition-colors shadow-lg inline-flex items-center gap-2"
          >
            <span>🧭</span> 여행 시작하기
          </button>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 bg-white">
        <div className="max-w-[1440px] mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl text-neutral-800 mb-4">
            왜 여떠잼을 선택해야 할까요?
          </h2>
          <p className="text-lg md:text-xl text-neutral-600 mb-16">
            여행의 모든 순간을 더욱 특별하게 만들어드립니다
          </p>

          <div className="grid lg:grid-cols-3 gap-12">
            {features.map(({ img, alt, title, desc }) => (
              <Feature key={title} img={img} alt={alt} title={title} desc={desc} />
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24 bg-neutral-50">
        <div className="max-w-[1440px] mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl text-neutral-800 mb-6">
            여행자들의 후기
          </h2>
          <p className="text-neutral-600 mb-6">실시간으로 올라오는 최신 후기</p>
          <button
            onClick={() => navigate("/reviews/new")}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-300 text-neutral-800 hover:bg-neutral-100 mb-10"
          >
            ✍️ 리뷰 작성하러 가기
          </button>

          <Testimonials limit={5} pollMs={30000} enableSSE={false} />
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 bg-neutral-800 text-center text-white px-6">
        <h2 className="text-4xl md:text-5xl mb-6">지금 바로 여행을 시작하세요</h2>
        <p className="text-xl md:text-2xl text-neutral-300 max-w-3xl mx-auto">
          수많은 여행자들이 이미 여떠잼과 함께 특별한 추억을 만들고 있습니다
        </p>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}

function Feature({ img, alt, title, desc }) {
  return (
    <div className="text-center p-8 rounded-xl hover:bg-neutral-50 transition-colors">
      <div className="w-20 h-20 bg-neutral-300 rounded-lg mx-auto mb-6 flex items-center justify-center">
        <img src={img} alt={alt} className="w-12 h-12 object-contain" draggable="false" />
      </div>
      <h3 className="text-2xl text-neutral-800 mb-4">{title}</h3>
      <p className="text-lg text-neutral-600 leading-relaxed">{desc}</p>
    </div>
  );
}

function Footer() {
  const sections = [
    { title: "서비스", items: ["여행 계획", "여행지 정보", "안전 서비스"] },
    { title: "고객지원", items: ["도움말", "문의하기", "FAQ"] },
  ];

  return (
    <footer className="bg-neutral-900 text-white py-16">
      <div className="max-w-[1440px] mx-auto px-6 grid lg:grid-cols-4 gap-12">
        <div>
          <div className="flex items-center mb-6">
            <span className="text-2xl">🗺 여떠잼</span>
          </div>
          <p className="text-lg text-neutral-400">당신의 완벽한 여행 파트너</p>
        </div>

        {sections.map(({ title, items }) => (
          <div key={title}>
            <h4 className="text-lg mb-6">{title}</h4>
            <ul className="space-y-3 text-neutral-400">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}

        <div>
          <h4 className="text-lg mb-6">팔로우</h4>
          <div className="flex flex-col space-y-2 text-neutral-400">
            <span className="hover:text-white cursor-pointer">Facebook</span>
            <span className="hover:text-white cursor-pointer">Instagram</span>
            <span className="hover:text-white cursor-pointer">Twitter</span>
          </div>
        </div>
      </div>

      <div className="border-t border-neutral-800 mt-12 pt-8 text-center text-lg text-neutral-400">
        © 2025 여떠잼. All rights reserved.
      </div>
    </footer>
  );
}
