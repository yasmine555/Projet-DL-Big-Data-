import React from 'react';

const conditions = [
  { value: 'hypertension', label: 'Hypertension' },
  { value: 'diabetes', label: 'Diabetes' },
  { value: 'heartDisease', label: 'Heart Disease' },
  { value: 'stroke', label: 'Previous Stroke' },
  { value: 'tbi', label: 'Traumatic Brain Injury' },
  { value: 'depression', label: 'Depression/Anxiety' },
  { value: 'sleepApnea', label: 'Sleep Apnea' },
  { value: 'thyroidDisorder', label: 'Thyroid Disorder' },
];

export default function MedicalHistorySection({ visible, userType, formData, onChange }) {
  if (!visible) return null;
  const set = (patch) => onChange({ ...formData, ...patch });
  const toggle = (value) => {
    const list = formData.existingConditions || [];
    set({ existingConditions: list.includes(value) ? list.filter(v => v !== value) : [...list, value] });
  };

  return (
    <div className="p-5">
      <div className="flex items-center mb-6 border-b pb-3">
        <div className="p-2 rounded-full mr-3 bg-gradient-to-r from-pink-100 to-red-100">
          <svg className="w-6 h-6 text-pink-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Medical History</h2>
      </div>

      <div>
        <label className="block text-gray-700 text-base font-medium mb-2">Existing Conditions</label>
        <div className="bg-white rounded-lg p-4 shadow border border-pink-100">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {conditions.map(opt => (
              <label key={opt.value} className="inline-flex items-center bg-pink-50 hover:bg-pink-100 transition p-2.5 rounded-lg cursor-pointer">
                <input
                  type="checkbox"
                  checked={(formData.existingConditions || []).includes(opt.value)}
                  onChange={() => toggle(opt.value)}
                  className="h-4 w-4 text-pink-600 mr-3"
                />
                <span className="text-gray-700">{opt.label}</span>
              </label>
            ))}
          </div>
        </div>


      </div>
    </div>
  );
}