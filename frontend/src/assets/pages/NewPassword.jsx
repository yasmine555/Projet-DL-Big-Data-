import React from "react";
import Navbar from "../components/navBar";
import Footer from "../components/footer";
import ResetPassword from "../components/newPassword";
import { useNavigate } from "react-router-dom";

export default function NewPassword() {
  const nav = useNavigate();
  return (
    <div className="min-h-screen relative">
      <div className="pointer-events-none absolute inset-0 bg-neutral-100" />
      <div className="relative z-10">
        <Navbar />
        <div className="container mx-auto px-6 pt-24 pb-16">
          <div className="mx-auto w-full max-w-md">
            <ResetPassword onSuccess={() => nav('/doctor/signin')} />
          </div>
        </div>
        <Footer />
      </div>
    </div>
  );
}
