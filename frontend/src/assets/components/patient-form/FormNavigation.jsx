import React from 'react';

export default function FormNavigation({ activeSection, sections, isLoading, userType, onPrev, onNext, onSubmit }) {
  return (
    <div className="p-5 bg-gradient-to-b from-transparent to-indigo-50 flex flex-col md:flex-row justify-between items-center">
      <div className="flex space-x-4 mb-4 md:mb-0">
        {activeSection !== sections[0].id && (
          <button type="button" onClick={onPrev} className="btn-secondary btn-sm flex items-center">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
            </svg>
            Previous
          </button>
        )}
        {activeSection !== sections[sections.length - 1].id && (
          <button type="button" onClick={onNext} className="btn-primary btn-sm flex items-center">
            Next
            <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
            </svg>
          </button>
        )}
      </div>
      {activeSection === 'mri' && (
        <button type="submit" onClick={onSubmit} disabled={isLoading} className={`btn-primary btn-sm flex items-center ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}>
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </>
          ) : (
            <>
              Submit for Analysis
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </>
          )}
        </button>
      )}
    </div>
  );
}