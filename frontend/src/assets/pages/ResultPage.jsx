import ResNav from "../components/results/navRes.jsx";
import Results from "../components/results/results.jsx";

export default function ResultPage() {
  return (
    <div className="flex flex-col h-[100vh]">
      <div className="pointer-events-none absolute inset-0 " />

      <ResNav />
      <main className="flex-grow pt-20">

        <Results />
      </main>
    </div>
  );
}