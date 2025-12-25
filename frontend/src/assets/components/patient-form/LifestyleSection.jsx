import React from 'react';

export default function LifestyleSection({ visible, formData, onChange }) {
  if (!visible) return null;
  const set = (patch) => onChange({ ...formData, ...patch });
  return (
    <div className="p-5">
      <div className="flex items-center mb-6 border-b pb-3">
        <div className="p-2 rounded-full mr-3 bg-gradient-to-r from-teal-100 to-green-100">
          <svg className="w-6 h-6 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Lifestyle Factors</h2>
      </div>

      <div className="grid grid-cols-1 gap-4">
        <div>
          <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="physicalActivity">Physical Activity Level</label>
          <select
            id="physicalActivity"
            value={formData.physicalActivity}
            onChange={(e) => set({ physicalActivity: e.target.value })}
            className="input px-1 text-base"
          >
            <option value="">Select activity level</option>
            <option value="sedentary">Sedentary (little to no exercise)</option>
            <option value="light">Light activity (1-3 days/week)</option>
            <option value="moderate">Moderate activity (3-5 days/week)</option>
            <option value="active">Very active (6-7 days/week)</option>
            <option value="veryActive">Extra active (professional athlete level)</option>
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="smokingStatus">Smoking Status</label>
            <select
              id="smokingStatus"
              value={formData.smokingStatus}
              onChange={(e) => set({ smokingStatus: e.target.value })}
              className="input px-1 text-base "
            >
              <option value="">Select smoking status</option>
              <option value="never">Never smoked</option>
              <option value="former">Former smoker</option>
              <option value="occasional">Occasional smoker</option>
              <option value="regular">Regular smoker</option>
            </select>
          </div>

          <div>
            <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="alcoholConsumption">Alcohol Consumption</label>
            <select
              id="alcoholConsumption"
              value={formData.alcoholConsumption}
              onChange={(e) => set({ alcoholConsumption: e.target.value })}
              className="input px-1 text-base"
            >
              <option value="">Select alcohol consumption</option>
              <option value="none">None</option>
              <option value="occasional">Occasional (1-2 drinks/week)</option>
              <option value="moderate">Moderate (3-7 drinks/week)</option>
              <option value="heavy">Heavy (8+ drinks/week)</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}