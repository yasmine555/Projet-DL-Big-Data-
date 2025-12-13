import React from "react";

export default function AdminUserItem({ user, onApprove, onDelete }) {
  const relUrl = user?.credential_file?.stored_url;
  const apiBase = process.env.REACT_APP_API_BASE || "http://localhost:8000";
  const certUrl = relUrl ? `${apiBase}${relUrl}` : null;
  return (
    <div className="rounded-lg border border-gray-200 bg-white/80 p-4 shadow-sm">
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4 items-start">
        <div className="md:col-span-4">
          <div className="font-medium text-gray-900">
            {user.first_name} {user.last_name}
            <span className="ml-2 text-xs text-gray-500">{user.email}</span>
          </div>
          <div className="text-sm text-gray-700">{user.institution} • {user.phone}</div>
          <div className="text-xs text-gray-500">Submitted: {user.created_at ? new Date(user.created_at).toLocaleString() : ""}</div>
          {user.credential_note && (
            <div className="mt-1 text-xs text-gray-700">Note: {user.credential_note}</div>
          )}
          <div className="mt-2">
            {user.status === "approved" && <span className="text-xs font-semibold text-green-700">Approved</span>}
            {user.status === "rejected" && <span className="text-xs font-semibold text-red-700">Rejected</span>}
            {user.status === "pending" && <span className="text-xs font-semibold text-yellow-700">Pending</span>}
          </div>
        </div>
        <div className="md:col-span-1">
          {certUrl && (
            <a className="block text-xs text-indigo-600 hover:text-indigo-700" href={certUrl} target="_blank" rel="noreferrer">View photo</a>
          )}
        </div>
        <div className="md:col-span-1 flex md:flex-col gap-2 justify-end md:justify-start">
          {user.status !== "approved" && (
            <button className="rounded-md bg-green-600 px-3 py-1.5 text-white hover:bg-green-700" onClick={()=>onApprove(user._id, true)}>
              Verify
            </button>
          )}
          {user.status !== "rejected" && (
            <button className="rounded-md bg-red-600 px-3 py-1.5 text-white hover:bg-red-700" onClick={()=>onApprove(user._id, false)}>
              Reject
            </button>
          )}
          {(user.status === "approved" || user.status === "rejected") && (
            <button className="rounded-md bg-gray-600 px-3 py-1.5 text-white hover:bg-gray-700" onClick={()=>onDelete(user._id)}>
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
