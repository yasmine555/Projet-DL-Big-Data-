// Frontend API client for FastAPI backend
// Aligns with backend routes in backend/app/api/endpoints/*.py

const BASE = (import.meta.env?.VITE_API_BASE_URL || "http://localhost:7000/api").replace(/\/$/, "");
const TOKEN_KEY = "token";

const withBase = (path) => `${BASE}${path.startsWith("/") ? "" : "/"}${path}`;

export const getToken = () => localStorage.getItem(TOKEN_KEY) || null;
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t || "");
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

function authHeaders(extra = {}) {
  const token = getToken();
  return token
    ? { ...extra, Authorization: `Bearer ${token}` }
    : { ...extra };
}

async function handle(res) {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const t = await res.text();
      msg = t || msg;
    } catch (_) {}
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

// -------- Auth --------
export async function login(formOrData) {
  const fd = formOrData instanceof FormData ? formOrData : new FormData();
  if (!(formOrData instanceof FormData)) {
    if (formOrData?.email) fd.append("email", formOrData.email);
    if (formOrData?.password) fd.append("password", formOrData.password);
  }
  const res = await fetch(withBase("/auth/login"), { method: "POST", body: fd });
  const data = await handle(res);
  if (data?.access_token) setToken(data.access_token);
  return data;
}

export async function signup(formOrData) {
  const fd = formOrData instanceof FormData ? formOrData : new FormData();
  if (!(formOrData instanceof FormData) && formOrData && typeof formOrData === "object") {
    for (const [k, v] of Object.entries(formOrData)) {
      if (v != null) fd.append(k, v);
    }
  }
  const res = await fetch(withBase("/auth/signup"), { method: "POST", body: fd });
  return handle(res); // { id, message, credential_file_url }
}

// Password helpers (optional)
export async function requestPasswordReset(email) {
  const res = await fetch(withBase("/auth/password/reset/request"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  return handle(res);
}

export async function confirmPasswordReset(token, new_password) {
  const res = await fetch(withBase("/auth/password/reset/confirm"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password }),
  });
  return handle(res);
}

// -------- Admin (requires admin Bearer token) --------
export async function adminListUsers() {
  const res = await fetch(withBase("/auth/admin/list"), { headers: authHeaders() });
  return handle(res); // { users: [...] }
}

export async function adminApproveUser(user_id, approve = true) {
  const res = await fetch(withBase("/auth/admin/approve"), {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id, approve }),
  });
  return handle(res); // { ok: true, status }
}

export async function adminDeleteUser(user_id) {
  const res = await fetch(withBase("/auth/admin/delete"), {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id }),
  });
  return handle(res); // { ok: true }
}

// -------- Doctor (requires doctor Bearer token) --------
export async function listPatients() {
  const res = await fetch(withBase("/doctor/patients"), { headers: authHeaders() });
  return handle(res); // PatientOut[]
}

export async function getPatientById(patientId) {
  const res = await fetch(withBase(`/doctor/patients/${encodeURIComponent(patientId)}`), {
    headers: authHeaders(),
  });
  return handle(res);
}

export async function createPatient(payload) {
  const res = await fetch(withBase("/doctor/patients"), {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload || {}),
  });
  return handle(res);
}

export async function updatePatient(patientId, payload) {
  const res = await fetch(withBase(`/doctor/patients/${encodeURIComponent(patientId)}`), {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload || {}),
  });
  return handle(res);
}

export async function deletePatient(patientId) {
  const res = await fetch(withBase(`/doctor/patients/${encodeURIComponent(patientId)}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return handle(res);
}

export async function getPatientConversations(patientId) {
  const res = await fetch(withBase(`/doctor/patients/${encodeURIComponent(patientId)}/conversations`), {
    headers: authHeaders(),
  });
  return handle(res); // { conversations: [...] }
}
// -------- MRI Scans (doctor) --------
export async function getPatientMRIScans(patientId) {
  const res = await fetch(withBase(`/mri/patient/${encodeURIComponent(patientId)}/scans`), {
    headers: authHeaders(),
  });
  return handle(res); // { scans: [...], count }
}

export async function analyzePatientMRI(patientId, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(withBase(`/mri/analyze-patient/${encodeURIComponent(patientId)}`), {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  return handle(res); // MRIAnalysisResponse
}

// -------- AI / Questionnaire --------
export async function askQuery(payload) {
  const res = await fetch(withBase("/ai/query/"), {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload || {}),
  });
  return handle(res); // EnhancedResponse
}

export async function getPatientResults(patientId) {
  // Backend returns list with latest result first (wrapped array) or []
  const res = await fetch(withBase(`/results/${encodeURIComponent(patientId)}`), {
    headers: authHeaders(),
  });
  return handle(res);
}
