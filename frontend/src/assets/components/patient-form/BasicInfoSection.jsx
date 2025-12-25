import React from 'react';

export default function BasicInfoSection({ visible, userType, formData, errors, onChange }) {
  if (!visible) return null;
  const set = (patch) => onChange({ ...formData, ...patch });
  return (
    <div className="p-5">
      <div className="flex items-center mb-6 border-b pb-3">
        <div className="p-2 rounded-full mr-3 bg-gradient-to-r from-blue-100 to-indigo-100">
          <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Basic Information</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {userType === 'doctor' && (
          <div className="md:col-span-2">
            <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="patientName">Patient Name</label>
            <input
              id="patientName"
              type="text"
              value={formData.patientName}
              onChange={(e) => set({ patientName: e.target.value })}
              className={`input input-sm w-70 ${errors.patientName ? 'border-red-300 focus:ring-red-300' : ''}`}
              placeholder="Enter patient full name"
            />
            {errors.patientName && <p className="text-red-500 text-sm mt-1">{errors.patientName}</p>}
          </div>
        )}

        <div>
          <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="patientAge">Age</label>
          <input
            id="patientAge"
            type="number"
            value={formData.patientAge}
            onChange={(e) => set({ patientAge: e.target.value })}
            min="0" max="120"
            className={`input input-sm w ${errors.patientAge ? 'border-red-300 focus:ring-red-300' : ''}`}
            placeholder="Enter age"
          />
          {errors.patientAge && <p className="text-red-500 text-sm mt-1">{errors.patientAge}</p>}
        </div>

        <div>
          <label className="block text-gray-700 text-base font-medium mb-2">Gender</label>
          <div className="grid grid-cols-2 gap-3">
            {['male','female'].map(g => (
              <label key={g} className={`bg-white border ${formData.patientGender === g ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'} rounded-lg p-3 flex items-center justify-center cursor-pointer`}>
                <input
                  type="radio"
                  name="patientGender"
                  value={g}
                  checked={formData.patientGender === g}
                  onChange={(e) => set({ patientGender: e.target.value })}
                  className="hidden"
                />
                <span className={`${formData.patientGender === g ? 'text-indigo-700 font-semibold' : 'text-gray-700'}`}>{g === 'male' ? 'Male' : 'Female'}</span>
              </label>
            ))}
          </div>
          {errors.patientGender && <p className="text-red-500 text-sm mt-1">{errors.patientGender}</p>}
        </div>

        <div>
          <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="educationLevel">Education Level</label>
          <select
            id="educationLevel"
            value={formData.educationLevel}
            onChange={(e) => set({ educationLevel: e.target.value })}
            className="input px-1 text-base w"
          >
            <option value="">Select education level</option>
            <option value="lessThanHighSchool">Less than high school</option>
            <option value="highSchool">High school</option>
            <option value="someCollege">Some college</option>
            <option value="bachelor">Bachelor's degree</option>
            <option value="graduate">Graduate degree</option>
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="flex items-center gap-3 bg-white rounded-lg p-3 cursor-pointer w-full">
            <input
              type="checkbox"
              checked={formData.familyHistory}
              onChange={(e) => set({ familyHistory: e.target.checked })}
              className="h-4 w-4 text-indigo-600"
            />
            <span className="text-gray-700">Family history of neurodegenerative diseases</span>
          </label>
        </div>
      </div>
    </div>
  );
}