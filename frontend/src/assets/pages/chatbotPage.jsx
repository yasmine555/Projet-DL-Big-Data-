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
    <div className="flex flex-col h-full m-0">
      <div className="pointer-events-none absolute inset-0 " />

      <ResNav />
      <main className="flex-grow pt-20 px-10">
        <div className="max-w-5xl mx-auto h-full py-6">
          <Chatbot containerClass="h-[calc(100vh-200px)]" />
        </div>
      </main>
    </div>
  );
}