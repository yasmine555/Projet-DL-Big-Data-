import React, { useState } from "react";
import { Link } from "react-router-dom";
import { requestPasswordReset } from "../../services/api";
import Toast from "./ui/Toast";

export default function ForgetPassword(){
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [ok, setOk] = useState(false);
  const [msg, setMsg] = useState("");
  const [toast, setToast] = useState({ show: false, type: "success", message: "" });

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true); setOk(false); setMsg("");
    try {
      await requestPasswordReset(email);
      setOk(true);
      setMsg("If the email exists, you will receive an email.");
      setToast({ show: true, type: "success", message: "Reset link sent if the email exists." });
    } catch (err) {
      setOk(false);
      setMsg("Unable to send reset email. Please try again.");
      setToast({ show: true, type: "danger", message: "Failed to send reset email." });
    } finally {
      setLoading(false);
    }
  };
return(

    <div className="flex flex-col shadow-lg bg-white justify-center px-6 py-12 lg:px-8 rounded-xl">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
       
        <h2 className="mt-10 text-center text-2xl tracking-tight text-gray-900">Forget your password?</h2>
        <h4>Enter your email and we'll send you a reset link.</h4>
     </div> 
    <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-md">
     <form onSubmit={submit} className="space-y-6">
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-gray-900">Email address</label>
        <div className="mt-2">
          <input id="email" type="email" name="email" value={email} onChange={(e)=>setEmail(e.target.value)} required autoComplete="email" className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
        </div>
      </div>

     

      <div>
        <button disabled={loading}  type="submit" className="mt-2 w-full rounded-md bg-indigo-600 px-4 py-2.5 text-white shadow hover:bg-indigo-700 disabled:opacity-60">{loading ? "Sending..." : "Send Reset Link"}</button>
   {toast.show && (
  <Toast type={toast.type} message={toast.message} onClose={() => setToast(prev => ({ ...prev, show: false }))} />
)}   
      </div>
 {msg && <p className="text-sm text-red-600">{msg}</p>}   
  
    </form>
    
    <div className="mt-2 text-center flex mt-5">
      <Link  className="text-indigo-600 flex-1 hover:text-indigo-700 no-underline font-semibold" to="/doctor/signin">Sign In</Link>
      <Link className="text-indigo-600 flex-1 hover:text-indigo-700 no-underline font-semibold" to="/doctor/signup">Sign Up</Link>
    </div>
  </div>
</div>

    
  );
  }