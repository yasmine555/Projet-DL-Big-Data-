import React from "react";

// Backend base URL for static files (uploads)
const BACKEND_BASE = (import.meta.env?.VITE_API_BASE_URL || "http://localhost:7000/api").replace(/\/api$/, "");

export default function AdminUserList({ users = [], onApprove, onDelete }) {
	if (!Array.isArray(users)) users = [];
	return (
		<div className="overflow-x-auto">
			<table className="min-w-full divide-y divide-gray-200">
				<thead className="bg-gray-50">
					<tr>
						<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
						<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
						<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Phone Number</th> 
						<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Credential File</th>
						<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
						<th className="px-4 py-2" />
					</tr>
				</thead>
				<tbody className="divide-y divide-gray-100 bg-white/60">
					{users.map((u) => {
						const id = u._id || u.id;
						// Build full URL for credential file
						const credentialUrl = u.credential_file?.stored_url
							? `${BACKEND_BASE}${u.credential_file.stored_url}`
							: null;
						return (
							<tr key={id}>
								<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">{u.first_name} {u.last_name}</td>
								<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-700">{u.email}</td>
								<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-700">{u.institution || "—"}</td>
								<td className="px-4 py-2 whitespace-nowrap text-sm text-center">
									{credentialUrl ? (
										<a
											href={credentialUrl}
											target="_blank"
											rel="noreferrer"
											className="text-indigo-600 hover:text-indigo-700"
											title="View credential file"
										>
											<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-6">
												<path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
											</svg>
										</a>
									) : (
										<span className="text-gray-400">—</span>
									)}
								</td>
								<td className="px-4 py-2 whitespace-nowrap text-sm">
									<span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${u.status === "approved" ? "bg-green-100 text-green-800" : (u.status === "rejected" ? "bg-red-100 text-red-800" : "bg-yellow-100 text-yellow-800")}`}>
										{u.status || "pending"}
									</span>
								</td>
								<td className="px-4 py-2 whitespace-nowrap text-right text-sm">
									<div className="flex gap-2 justify-end">
										<button onClick={() => onApprove?.(id, true)} className="rounded-md bg-indigo-600 px-3 py-1.5 text-white hover:bg-indigo-700">Approve</button>
										<button onClick={() => onApprove?.(id, false)} className="rounded-md bg-amber-500 px-3 py-1.5 text-white hover:bg-amber-600">Reject</button>
										<button onClick={() => onDelete?.(id)} className="rounded-md bg-red-600 px-3 py-1.5 text-white hover:bg-red-700">Delete</button>
									</div>
								</td>
							</tr>
						);
					})}
					{users.length === 0 && (
						<tr>
							<td className="px-4 py-6 text-center text-sm text-gray-500" colSpan={5}>No users found.</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
}

