import React, { useState } from "react";
import { login } from "../../services/api";
import Toast from "../components/ui/Toast";


export default function AdminSignInForm({ onSuccess }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState(null);
  const [toast, setToast] = useState({ show: false, type: "success", message: "" });
  const [showPassword, setShowPassword] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true); setMsg(null);
    setToast({ show: false, type: "success", message: "" });
  
    try {
      const fd = new FormData();
      fd.append("email", email);
      fd.append("password", password);
      await login(fd);
      setToast({ show: true, type: "success", message: "Admin login successful!" });
    
      onSuccess?.();

    } catch (err) {
      const message = "Invalid credentials or not an admin account.";
      setMsg(message);
      setToast({ show: true, type: "danger", message });
    } finally {
      setLoading(false);
    }
  };

  return (
     <div className="flex my-20 min-h-full shadow-lg/30 bg-white  flex-col justify-center px-6 py-12 lg:px-8 rounded-xl">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
       
        <h2 className="mt-10 text-center text-2xl font-bold tracking-tight text-gray-900">Admin Login</h2>
     </div> 
     <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
   
    <form onSubmit={submit} className="space-y-6">
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-gray-900">Email address</label>
        <div className="mt-2">
          <input id="email" type="email" name="email" value={email} onChange={(e)=>setEmail(e.target.value)} required autoComplete="email" className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between">
          <label htmlFor="password" className="block text-sm font-medium text-gray-900">Password</label>
        </div>
        <div className="mt-2">
          <input id="password" type={showPassword ? "text" : "password"} name="password" value={password} required autoComplete="current-password" onChange={(e)=>setPassword(e.target.value)}  className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
        </div>
      </div>

      <div>
        <button disabled={loading}  type="submit" className="mt-2 w-full rounded-md bg-indigo-600 px-4 py-2.5 text-white shadow hover:bg-indigo-700 disabled:opacity-60">{loading ? "Signing in..." : "Sign In"}</button>
           {toast.show && (
              <Toast type={toast.type} message={toast.message} onClose={() => setToast(prev => ({ ...prev, show: false }))} />
            )}
      </div>

 {msg && <p className="text-sm text-red-600">{msg}</p>}   
  
    </form>
    </div>
    </div>
  );
}
