import About from "@/components/About";
import Footer from "@/components/Footer";
import { Cards } from "@/components/GridCards";
import Header from "@/components/Header";
import Home from "@/components/Home";
import SmoothScroll from "@/components/ui/Scrolling";

export default function Twisense() {
  return (
    <SmoothScroll>
      <main className="min-h-screen bg-[#101010] text-white">
        <Header text="Twisense"></Header>
        <div id="home">
          <Home />
        </div>
        <div id="features">
          <Cards />
        </div>
        <div id="about">
          <About direction="down" />
        </div>
        <Footer />
      </main>
    </SmoothScroll>
  );
}
