import React from 'react';

export default function TestsSection({ visible, formData, onChange }) {
  if (!visible) return null;
  const set = (patch) => onChange({ ...formData, ...patch });
  return (
    <div className="p-5">
      <div className="flex items-center mb-6 border-b pb-3">
        <div className="p-2 rounded-full mr-3 bg-gradient-to-r from-yellow-100 to-orange-100">
          <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Clinical Tests & Biomarkers</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg p-4 shadow border border-orange-100">
          <h3 className="font-bold text-gray-700 mb-3">Cognitive Assessment Scores</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-gray-600 font-medium mb-2" htmlFor="mmse">MMSE Score (Normal: 24-30)</label>
              <input
                id="mmse"
                type="number"
                value={formData.mmse}
                onChange={(e) => set({ mmse: e.target.value })}
                min="0" max="30"
                className="input input-sm"
                placeholder="e.g. 24"
              />
            </div>
            <div>
              <label className="block text-gray-600 font-medium mb-2" htmlFor="moca">MoCA Score (Normal: 26-30)</label>
              <input
                id="moca"
                type="number"
                value={formData.moca}
                onChange={(e) => set({ moca: e.target.value })}
                min="0" max="30"
                className="input input-sm"
                placeholder="e.g. 26"
              />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-4 shadow border border-orange-100">
          <h3 className="font-bold text-gray-700 mb-3">Biomarkers</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-gray-600 font-medium mb-2" htmlFor="amyloid">Amyloid Beta</label>
              <select
                id="amyloid"
                value={formData.amyloid}
                onChange={(e) => set({ amyloid: e.target.value })}
                className="input input-sm"
              >
                <option value="unknown">Unknown / Not Tested</option>
                <option value="negative">Negative (Normal)</option>
                <option value="positive">Positive (Abnormal)</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-600 font-medium mb-2" htmlFor="tau">Tau Protein</label>
              <select
                id="tau"
                value={formData.tau}
                onChange={(e) => set({ tau: e.target.value })}
                className="input input-sm"
              >
                <option value="unknown">Unknown / Not Tested</option>
                <option value="negative">Negative (Normal)</option>
                <option value="positive">Positive (Abnormal)</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-600 font-medium mb-2" htmlFor="apoe4">APOE4 Status</label>
              <select
                id="apoe4"
                value={formData.apoe4}
                onChange={(e) => set({ apoe4: e.target.value })}
                className="input input-sm"
              >
                <option value="unknown">Unknown / Not Tested</option>
                <option value="negative">Negative (Non-carrier)</option>
                <option value="positive">Positive (Carrier)</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}