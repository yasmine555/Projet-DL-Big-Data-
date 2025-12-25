import React, { useState } from "react";
import Navbar from "../components/navBar";
import Footer from "../components/footer";
import ForgetPassword from "../components/forgetPassword";


export default function ForgotPasswordRequest() {
  return (
    <div className="min-h-screen relative">
      <div className="pointer-events-none absolute inset-0 bg-neutral-100" />
      <div className="relative z-10">
        <Navbar />
        <div className="container mx-auto px-6 pt-24 pb-16">
          <div className="mx-auto w-full max-w-md">
            <ForgetPassword />
          </div>
        </div>
        <Footer />
      </div>
    </div>
   
  );
}
