import React from 'react';
export default function MRIUploadSection({ visible, errors, selectedFile, previewUrl, onFileChange, onPreviewChange, scans = [] }) {
  if (!visible) return null;
  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileChange(file);
      const reader = new FileReader();
      reader.onload = () => onPreviewChange(reader.result);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="p-5">
      <div className="flex items-center mb-6 border-b pb-3">
        <div className="p-2 rounded-full mr-3 bg-gradient-to-r from-indigo-100 to-blue-100">
          <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Brain MRI Upload</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
        <div>
          <label className="block text-gray-700 text-base font-medium mb-2" htmlFor="mriUpload">Upload MRI Scan</label>
          <div className={`border-2 border-dashed rounded-lg p-5 text-center ${errors.file ? 'border-red-300 bg-red-50' : 'border-indigo-200 bg-indigo-50'} hover:bg-indigo-100 transition cursor-pointer`}>
            <input
              id="mriUpload"
              type="file"
              accept=".jpg,.jpeg,.png"
              onChange={handleFile}
              className="hidden"
            />
            <label htmlFor="mriUpload" className="cursor-pointer flex flex-col items-center">
              <svg className="w-12 h-12 text-indigo-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="font-medium text-indigo-600">
                {selectedFile ? selectedFile.name : 'Click to upload MRI scan (JPG, PNG)'}
              </span>
            </label>
          </div>
          {errors.file && <p className="text-red-500 text-sm mt-1">{errors.file}</p>}
        </div>

        <div className="flex justify-center">
          {previewUrl ? (
            <div className="border-4 border-indigo-200 rounded-lg overflow-hidden shadow h-56 w-full">
              <img src={previewUrl} alt="MRI Preview" className="w-full h-full object-contain" />
            </div>
          ) : (
            <div className="border-4 border-indigo-100 rounded-lg p-6 h-56 w-full flex flex-col items-center justify-center bg-indigo-50">
              <svg className="w-12 h-12 text-indigo-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-indigo-400 text-center">MRI preview will appear here</p>
            </div>
          )}
        </div>
      </div>
      {Array.isArray(scans) && scans.length > 0 && (
        <div className="mt-2">
          <div className="text-xs text-gray-600 mb-1">Latest MRI scans</div>
          <ul className="space-y-1">
            {scans.slice(0, 5).map((s, i) => (
              <li key={s.id || i} className="text-xs">
                <a href={s.image_url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                  {new Date(s.created_at || Date.now()).toLocaleString()} — {s.prediction_class || 'unknown'}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}