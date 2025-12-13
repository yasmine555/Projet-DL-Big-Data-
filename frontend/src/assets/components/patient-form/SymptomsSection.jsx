import React from 'react';

const cognitive = [
  { value: 'memoryLoss', label: 'Memory loss' },
  { value: 'difficultyConcentrating', label: 'Difficulty concentrating' },
  { value: 'confusionDisorientation', label: 'Confusion/disorientation' },
  { value: 'difficultyPlanning', label: 'Difficulty planning or problem solving' },
  { value: 'languageProblems', label: 'Language or speech problems' },
];

const motor = [
  { value: 'tremors', label: 'Tremors' },
  { value: 'slowMovement', label: 'Slow movement' },
  { value: 'balanceProblems', label: 'Balance problems' },
  { value: 'coordinationIssues', label: 'Coordination issues' },

];

export default function SymptomsSection({ visible, formData, onChange }) {
  if (!visible) return null;
  const set = (patch) => onChange({ ...formData, ...patch });
  const toggle = (listKey, value) => {
    const list = formData[listKey] || [];
    set({ [listKey]: list.includes(value) ? list.filter(v => v !== value) : [...list, value] });
  };

  return (
    <div className="p-5">
      <div className="flex items-center mb-6 border-b pb-3">
        <div className="p-2 rounded-full mr-3 bg-gradient-to-r from-purple-100 to-indigo-100">
          <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Symptoms & Health</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-gray-700 text-base font-medium mb-2">Cognitive Symptoms</label>
          <div className="bg-white rounded-lg p-4 shadow border border-indigo-100">
            <div className="grid grid-cols-1 gap-3">
              {cognitive.map(opt => (
                <label key={opt.value} className="inline-flex items-center hover:bg-indigo-100 transition p-1 rounded-lg cursor-pointer">
                  <input
                    type="checkbox"
                    checked={(formData.cognitiveSymptoms || []).includes(opt.value)}
                    onChange={() => toggle('cognitiveSymptoms', opt.value)}
                    className="h-3 w-5 text-indigo-600 mr-3"
                  />
                  <span className="text-gray-700 ">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div>
          <label className="block text-gray-700 text-base font-medium mb-2">Motor Symptoms</label>
          <div className="bg-white rounded-lg p-4 shadow border border-indigo-100">
            <div className="grid grid-cols-1 gap-3">
              {motor.map(opt => (
                <label key={opt.value} className="inline-flex items-center  hover:bg-indigo-100 transition p-1 rounded-lg cursor-pointer">
                  <input
                    type="checkbox"
                    checked={(formData.motorSymptoms || []).includes(opt.value)}
                    onChange={() => toggle('motorSymptoms', opt.value)}
                    className="h-3 w-5 text-indigo-600 mr-3"
                  />
                  <span className="text-gray-700 ">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="md:col-span-2">
          <label className="bg-white rounded-lg p-4 shadow border border-indigo-100 flex items-center hover:bg-indigo-50 transition cursor-pointer w-90">
            <input
              type="checkbox"
              checked={formData.sleepIssues}
              onChange={(e) => set({ sleepIssues: e.target.checked })}
              className="h-5 w-5 text-indigo-600 mr-3"
            />
            <span className=" text-gray-700">Do you experience sleep disturbances?</span>
          </label>
        </div>
      </div>
    </div>
  );
}