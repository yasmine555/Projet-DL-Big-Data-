import React, { useState } from "react";
import { signup } from "../../services/api";
import { Link } from "react-router-dom";

import Toast from "./ui/Toast";

function Register({ onSuccess }) {
  const [form, setForm] = useState({
    first_name: "",
    last_name: "",
    email: "",
    phone: "",

    password: "",
    confirm_password: "",
  });
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState({ show: false, type: "success", message: "" });

  const onFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    if (f) {
      if (f.type.startsWith("image/")) {
        const url = URL.createObjectURL(f);
        setPreviewUrl(url);
      } else {
        setPreviewUrl(null);
      }
    } else {
      setPreviewUrl(null);
    }
  };

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const submit = async (e) => {
    e.preventDefault();
    if ((form.password || "") !== (form.confirm_password || "")) {
      setToast({ show: true, type: "danger", message: "Passwords do not match." });
      return;
    }
    setLoading(true);
    setToast({ show: false, type: "success", message: "" });
    setErrors({});
    try {
      const formData = new FormData();
      formData.append("first_name", form.first_name);
      formData.append("last_name", form.last_name);
      formData.append("email", form.email);
      formData.append("phone", form.phone);
      formData.append("password", form.password);
      if (file) formData.append("credential_file", file);

      await signup(formData);
      setToast({ show: true, type: "success", message: "Registration successful!" });
      // Clear form on success
      setForm({ first_name: "", last_name: "", email: "", phone: "", password: "", confirm_password: "" });
      setFile(null);
      setPreviewUrl(null);
      if (onSuccess) onSuccess();
    } catch (err) {
      // Extract meaningful server message when possible
      let msg = "Registration failed.";
      const data = err?.response?.data;
      if (typeof data === "string") msg = data;
      else if (data?.detail) msg = Array.isArray(data.detail) ? data.detail.map(d => d.msg || d).join("; ") : (data.detail.msg || data.detail);
      else if (data?.error) msg = data.error;
      else if (data?.message) msg = data.message;
      // Common causes to surface
      if (/email/i.test(String(msg)) && /exist|taken|already/i.test(String(msg))) {
        msg = "Email already exists. Please use another email.";
      }
      setToast({ show: true, type: "danger", message: msg });
      setErrors(data || {});
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex my-20 min-h-full w-100 shadow-lg/30 bg-white flex-col justify-center px-6 py-12 lg:px-8 rounded-xl">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <h2 className="mt-10 text-center text-2xl font-bold tracking-tight text-gray-900">Doctor Register</h2>
      </div>
      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
        <form onSubmit={submit} className="space-y-6 pr-4">
          <div>
            <label htmlFor="credential-file" className="block text-sm font-medium text-gray-900">Identity Proof * </label>
            <p>Please submit a valid identity proof document (Dimploma/Certification..).</p>
            <div className="mt-2 flex flex-col items-center justify-center rounded-md border border-dashed border-gray-300 px-6 py-6">
              <div className="text-center">
                <div className="mt-4 flex text-sm text-gray-600 items-center gap-2">
                  <label htmlFor="credential-file" className="relative cursor-pointer rounded-md bg-transparent font-semibold text-indigo-600 focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-indigo-500 hover:text-indigo-500">
                    <span>Upload a file *</span>
                    <input id="credential-file" type="file" name="credential_file" className="sr-only" required onChange={onFile} />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                {previewUrl && (
                  <div className="mt-3">
                    <img src={previewUrl} alt="Preview" className="mx-auto max-h-24 rounded" />
                  </div>
                )}
              </div>
            </div>
          </div>
          <div>
            <label htmlFor="first_name" className="block text-sm font-medium text-gray-900">First name * </label>
            <div className="mt-2">
              <input id="first_name" type="text" name="first_name" autoComplete="given-name" required className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" value={form.first_name} onChange={onChange} />
            </div>
          </div>
          <div>
            <label htmlFor="last_name" className="block text-sm font-medium text-gray-900">Last name * </label>
            <div className="mt-2">
              <input id="last_name" type="text" name="last_name" autoComplete="family-name" required className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" value={form.last_name} onChange={onChange} />
            </div>
          </div>
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-900">Email address *</label>
            <div className="mt-2">
              <input id="email" type="email" name="email" autoComplete="email" required className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" value={form.email} onChange={onChange} />
            </div>
          </div>
          <div>
            <label htmlFor="phone" className="block text-sm font-medium text-gray-900">Phone number *</label>
            <div className="mt-2">
              <input id="phone" type="text" name="phone" className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" placeholder="123-456-7890" value={form.phone} onChange={onChange} />
            </div>
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-900">Password *</label>
            <div className="mt-2">
              <input id="password" type="password" name="password" required className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" value={form.password} onChange={onChange} />
            </div>
          </div>
          <div>
            <label htmlFor="confirm_password" className="block text-sm font-medium text-gray-900">Confirm Password *</label>
            <div className="mt-2">
              <input id="confirm_password" type="password" name="confirm_password" required className="block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 border border-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-600" value={form.confirm_password} onChange={onChange} />
            </div>
          </div>
          <div className="sm:col-span-6">
            <button
              disabled={loading}
              className="w-full cursor-pointer rounded-md bg-indigo-600 px-4 py-2.5 text-white shadow hover:bg-indigo-700 disabled:opacity-60"
            >
              {loading ? "Submitting…" : "Sign Up"}
            </button>
            {toast.show && (
              <Toast type={toast.type} message={toast.message} onClose={() => setToast(prev => ({ ...prev, show: false }))} />
            )}
          </div>

        </form>
        <p className="mt-10 text-center text-sm text-gray-500">Aldready have an account?</p>
        <div className="mt-2 text-center">
          <Link className="text-indigo-600 hover:text-indigo-700 no-underline font-semibold" to="/doctor/signin">Sign In</Link>
        </div>
      </div>

    </div>
  );
}

export default Register;