import React, { useEffect, useState } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { confirmPasswordReset } from "../../services/api";
import Toast from "./ui/Toast";

function ResetPassword() {
  const [search] = useSearchParams();
  const [token, setToken] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [ok, setOk] = useState(false);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const [toast, setToast] = useState({ show: false, type: "success", message: "" });
  const navigate = useNavigate();

   useEffect(() => {
      const t = search.get("token");
      if (t) setToken(t);
    }, [search]);

  const submit = async (e) => {
    e.preventDefault();
    if (!password) { setMsg("New password is required"); setToast({ show: true, type: "danger", message: "New password is required" }); return; }
    if (password !== confirm) { setMsg("Passwords do not match"); setToast({ show: true, type: "danger", message: "Passwords do not match" }); return; }
    setLoading(true); setOk(false); setMsg("");
    try {
      await confirmPasswordReset(token, password);
      setOk(true);
      setMsg("Password updated. Redirecting to sign in…");
      setToast({ show: true, type: "success", message: "Password updated successfully." });
      setTimeout(() => navigate("/doctor/signin"), 1200);
    } catch (err) {
      setOk(false);
      setMsg("Failed to reset password.");
      setToast({ show: true, type: "danger", message: "Failed to reset password." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex my-20 min-h-full shadow-lg bg-white  flex-col justify-center px-6 py-12 lg:px-8 rounded-xl">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
       
        <h2 className="mt-10 text-center text-2xl font-bold tracking-tight text-gray-900">Reset Your Password</h2>
     </div> 
     <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
     <form onSubmit={submit} className="space-y-6">
      <div>
        <label htmlFor="password1" className="block text-sm font-medium text-gray-900">Password</label>
        <div className="mt-2">
          <input id="password1" type="password" placeholder="New password"  value={password} onChange={(e)=>setPassword(e.target.value)} required className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between">
          <label htmlFor="password" className="block text-sm font-medium text-gray-900">Confirm Password</label>
        </div>
        <div className="mt-2">
          <input id="password" type="password" name="password_confirm" placeholder="Confirm new password" value={confirm} onChange={(e)=>setConfirm(e.target.value)} required  className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
        </div>
      </div>

      <div>
        <button disabled={loading}  type="submit" className="mt-2 w-full rounded-md bg-indigo-600 px-4 py-2.5 text-white shadow hover:bg-indigo-700 disabled:opacity-60">{loading ? "Resetting..." : "Resete password"}</button>
      </div>
 {msg && <p className="text-sm text-red-600">{msg}</p>}   
  
    </form>
    <p className="mt-10 text-center text-sm text-gray-500">Back to</p>
    <div className="mt-2 text-center">
      <Link className="text-indigo-600 hover:text-indigo-700 no-underline font-semibold" to="/doctor/signin">Sign In</Link>
    </div>
  </div>
{toast.show && (
  <Toast type={toast.type} message={toast.message} onClose={() => setToast(prev => ({ ...prev, show: false }))} />
)}
</div>
    
  );
}
export default ResetPassword;