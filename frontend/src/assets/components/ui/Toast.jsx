import React, { useState } from "react";
function Toast({ type = "success", message = "", onClose, duration = 8000 }) {
    const [isVisible, setIsVisible] = React.useState(false);
    const [progress, setProgress] = React.useState(100);

    React.useEffect(() => {
        // Trigger fade-in animation
        setIsVisible(true);

        // Auto-dismiss timer
        const dismissTimer = setTimeout(() => {
            handleClose();
        }, duration);

        // Progress bar animation
        const progressInterval = setInterval(() => {
            setProgress((prev) => {
                const decrement = (100 / duration) * 50; // Update every 50ms
                return Math.max(0, prev - decrement);
            });
        }, 50);

        return () => {
            clearTimeout(dismissTimer);
            clearInterval(progressInterval);
        };
    }, [duration]);

    const handleClose = () => {
        setIsVisible(false);
        setTimeout(() => {
            onClose();
        }, 300); // Wait for fade-out animation
    };

    const styles = {
        success: {
            iconBg: "bg-green-100 text-green-600",
            border: "border-green-200",
            bodyBg: "bg-white",
            progressBg: "bg-green-500",
        },
        danger: {
            iconBg: "bg-red-100 text-red-600",
            border: "border-red-200",
            bodyBg: "bg-white",
            progressBg: "bg-red-500",
        },
        warning: {
            iconBg: "bg-yellow-100 text-yellow-600",
            border: "border-yellow-200",
            bodyBg: "bg-white",
            progressBg: "bg-yellow-500",
        },
    }[type] || {
        iconBg: "bg-blue-100 text-blue-600",
        border: "border-blue-200",
        bodyBg: "bg-white",
        progressBg: "bg-blue-500",
    };

    return (
        <div
            className={`fixed top-4 right-4 z-50 w-full max-w-xs max-h-50 overflow-hidden ${styles.bodyBg} rounded-lg shadow-lg border ${styles.border} transition-all duration-500 ease-in-out transform ${isVisible ? "translate-x-0 opacity-100" : "translate-x-full opacity-0"
                }`}
            role="alert"
        >
            <div className="flex items-center p-4">
                <div className={`inline-flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-lg ${styles.iconBg}`}>
                    {type === "success" && (
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                clipRule="evenodd"
                            />
                        </svg>
                    )}
                    {type === "danger" && (
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                                clipRule="evenodd"
                            />
                        </svg>
                    )}
                    {type === "warning" && (
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                                clipRule="evenodd"
                            />
                        </svg>
                    )}
                    <span className="sr-only">{type}</span>
                </div>
                <div className="ml-3 text-sm font-medium text-gray-900">{message}</div>
                <button
                    type="button"
                    onClick={handleClose}
                    className="ml-auto -mx-1.5 -my-1.5 bg-white text-gray-400 hover:text-gray-900 rounded-lg focus:ring-2 focus:ring-gray-300 p-1.5 hover:bg-gray-100 inline-flex h-8 w-8 items-center justify-center"
                    aria-label="Close"
                >
                    <span className="sr-only">Close</span>
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path
                            fillRule="evenodd"
                            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                            clipRule="evenodd"
                        />
                    </svg>
                </button>
            </div>
            {/* Progress bar */}
            <div className="h-1 bg-gray-200">
                <div
                    className={`h-full ${styles.progressBg} transition-all duration-50 ease-linear`}
                    style={{ width: `${progress}%` }}
                />
            </div>
        </div>
    );
}
export default Toast;