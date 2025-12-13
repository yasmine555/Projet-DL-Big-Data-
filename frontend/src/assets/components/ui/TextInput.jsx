import React from "react";

export default function TextInput({
  icon: Icon,
  type = "text",
  name,
  value,
  onChange,
  placeholder,
  error,
  required,
  autoComplete
}) {
  return (
    <div className="w-full">
      <div className={`flex items-center gap-2 rounded-md border ${error ? "border-red-400 ring-1 ring-red-400" : "border-gray-200"} bg-white/90 px-3 py-2 shadow-sm focus-within:ring-2 focus-within:ring-indigo-500`}>
        {Icon ? <Icon className="h-5 w-5 text-gray-400" /> : null}
        <input
          className="w-full bg-transparent outline-none placeholder:text-gray-400"
          type={type}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          required={required}
          autoComplete={autoComplete}
        />
      </div>
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
}
