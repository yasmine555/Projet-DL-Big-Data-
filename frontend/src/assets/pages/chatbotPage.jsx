import { useEffect } from "react";
import { useParams } from "react-router-dom";
import ResNav from "../components/results/navRes.jsx";
import Chatbot from "../components/AI Assistant/chatbot.jsx";

export default function ChatbotPage() {
  const { patientId } = useParams();

  useEffect(() => {
    // Ensure patient ID is available in sessionStorage for the chatbot
    if (patientId) {
      sessionStorage.setItem('currentPatientId', patientId);
      sessionStorage.setItem('patient_id', patientId);
    }
  }, [patientId]);

  return (
    <div className="flex flex-col h-[100vh]">
      <div className="pointer-events-none absolute inset-0 bg-neutral-100" />

      <ResNav />
      <main className="flex-grow pt-20 px-10">
        <div className="max-w-5xl mx-auto h-full py-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">AI Medical Assistant</h2>
          <Chatbot containerClass="h-[calc(100vh-200px)]" />
        </div>
      </main>
    </div>
  );
}