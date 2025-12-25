import React, { useState } from "react";

import { HiEnvelope, HiLockClosed } from "react-icons/hi2";
import icon from '../image/icon.png';
import { Link } from "react-router-dom";
import { login } from "../../services/api";
import Toast from "./ui/Toast";

function DoctorSignInForm({ onSuccess }) {
  const [form, setForm] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState(null);
  const [toast, setToast] = useState({ show: false, type: "success", message: "" });

  const update = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    if (errors[e.target.name]) setErrors({ ...errors, [e.target.name]: null });
  };

  const validate = () => {
    const e = {};
    if (!form.email) e.email = "Email is required";
    if (!form.password) e.password = "Password is required";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const submit = async (e) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true); setMsg(null);
    try {
      const fd = new FormData();
      fd.append("email", form.email);
      fd.append("password", form.password);
      await login(fd);
      localStorage.setItem("userType", "doctor"); // Explicitly set user role
      setToast({ show: true, type: "success", message: "Signed in successfully." });
      onSuccess?.();
    } catch (err) {
      setMsg("Invalid email or password, or account not approved.");
      setToast({ show: true, type: "danger", message: "Invalid credentials or not approved." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex my-20 min-h-full shadow-lg/30 bg-white  flex-col justify-center px-6 py-12 lg:px-8 rounded-xl">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">

        <h2 className="mt-10 text-center text-2xl font-bold tracking-tight text-gray-900">Sign in to your account</h2>
      </div>
      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
        <form onSubmit={submit} className="space-y-6  pr-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-900">Email address</label>
            <div className="mt-2">
              <input id="email" type="email" name="email" value={form.email} onChange={update} required autoComplete="email" className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="password" className="block text-sm font-medium text-gray-900">Password</label>
              <div className="flex justify-end">
                <Link className="text-indigo-600 hover:text-indigo-700 no-underline font-semibold" to="/doctor/forgot-password">Forget my password</Link>

              </div>
            </div>
            <div className="mt-2">
              <input id="password" type="password" name="password" value={form.password} onChange={update} required autoComplete="current-password" className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" />
            </div>
          </div>

          <div>
            <button disabled={loading} type="submit" className="mt-2 w-full rounded-md bg-indigo-600 px-4 py-2.5 text-white shadow hover:bg-indigo-700 disabled:opacity-60">{loading ? "Signing in..." : "Sign In"}</button>
            {toast.show && (
              <Toast type={toast.type} message={toast.message} onClose={() => setToast(prev => ({ ...prev, show: false }))} />
            )}
          </div>
          {msg && <p className="text-sm text-red-600">{msg}</p>}

        </form>
        <p className="mt-10 text-center text-sm text-gray-500">You are new here?</p>
        <div className="mt-2 text-center">
          <Link className="text-indigo-600 hover:text-indigo-700 no-underline font-semibold" to="/doctor/signup">Sign Up</Link>
        </div>
      </div>
    </div>


  );
}

export default DoctorSignInForm;