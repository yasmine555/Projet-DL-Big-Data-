import React from "react";

export default function FileInput({
  name,
  onChange,
  accept,
  icon: Icon,
  hint,
  error,
  required
}) {
  return (
    <div className="w-full">
      <label className={`flex items-center gap-2 cursor-pointer rounded-md border ${error ? "border-red-400 ring-1 ring-red-400" : "border-none"} bg-white/90 px-3 py-2 shadow-sm hover:bg-white`}>
        {Icon ? <Icon className="h-5 w-5 text-gray-400" /> : null}
        <span className="text-sm text-gray-700 flex-1 truncate">{hint || "Choose a file"}</span>
        <input
          type="file"
          name={name}
          accept={accept}
          onChange={onChange}
          required={required}
          className="hidden"
        />
      </label>
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
}
