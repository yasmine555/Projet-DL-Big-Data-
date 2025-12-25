import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AdminSignInForm from "./adminSignin.jsx";

function AdminLogin() {
     const nav = useNavigate();
    return(
        <div className="min-h-screen relative">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-100" />
      <div className="relative z-10">
        
        <div className="container mx-auto px-6 pt-24 pb-16">
          <div className="mx-auto w-full max-w-md">
         
        <AdminSignInForm onSuccess={() => nav('/admin/dashboard')} />
            </div>
            </div>
            </div>
            </div>

    );
}

export default AdminLogin;