import React, { useEffect, useState } from "react";
import Navbar from "../components/navBar";
import AdminUserList from "./userList.jsx";
import { adminListUsers, adminApproveUser, adminDeleteUser } from "../../services/api";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/doctor/navDoc.jsx";


export default function AdminDashboard() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const nav = useNavigate();

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const data = await adminListUsers();
      setUsers(data.users || []);
    } catch (err) {
      setError("Unauthorized or failed to load.");
      // if unauthorized, go to admin signin
      setTimeout(()=> nav('/admin/login'), 800);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const onApprove = async (user_id, approve) => {
    try {
      await adminApproveUser(user_id, approve);
      await load();
    } catch (err) {
      // noop
    }
  };

  const onDelete = async (user_id) => {
    try {
      await adminDeleteUser(user_id);
      await load();
    } catch (err) {
      // noop
    }
  };
  

  return (
    <div className="min-h-screen relative">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-100" />
      <Navbar onSignOut={() => { localStorage.removeItem('token'); navigate('/admin/login'); }}/>
      <div className="relative container mx-auto px-6 pt-28 min-h-[calc(100vh-96px)]">
        <div className="rounded-2xl bg-white/80 p-6 shadow-xl ring-1 ring-black/5">
          <div className="mb-4 flex items-center justify-between">
            <h1 className="text-2xl font-semibold text-gray-900">Admin Dashboard</h1>
            <button onClick={load} className="rounded-md bg-indigo-600 px-3 py-1.5 text-white hover:bg-indigo-700">Refresh</button>
          </div>
           
          {loading && <p className="text-sm text-gray-600">Loading…</p>}
          {error && <p className="text-sm text-red-600">{error}</p>}
          {!loading && !error && (
            <AdminUserList users={users} onApprove={onApprove} onDelete={onDelete} />

          )}
        </div>
      </div>
      
    </div>
  );
}
