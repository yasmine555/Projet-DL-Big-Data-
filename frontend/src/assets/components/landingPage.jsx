import React from 'react';
import { useNavigate } from 'react-router-dom';
import dl from '../image/deepl.png';
import xai from '../image/xai.png';

function LandingSection() {
  const navigate = useNavigate();


  const handleUserTypeSelect = (userType) => {
    localStorage.setItem('userType', userType);
    if (userType === 'doctor') {
      navigate('/doctor/signin');
    } else {
      navigate('/patient/form');
    }
  };

  return (
    <div className="relative">
      {/* Fixed background */}
      <div
        className="fixed inset-0 bg-gradient-to-br from-blue-50 to-indigo-50"
        style={{ zIndex: -1 }}
      ></div>

      {/* Fixed Navigation */}


      {/* Content that scrolls (with padding-top to account for fixed navbar) */}
      <div className="relative min-h-screen pt-20">
        {/* Hero Section with more spacing */}
        <div className="container mx-auto px-6 py-16">
          {/* Centered title with more prominence */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
              Welcome to <span className="text-blue-500">NeuroAI</span> Assistant
            </h1>

            {/* Three words with more spacing and centered */}
            <div className="flex flex-wrap justify-center gap-4 mb-8">
              <div className="px-6 py-2 bg-blue-100 rounded-full text-blue-600 font-medium">Intelligent</div>
              <div className="px-6 py-2 bg-teal-100 rounded-full text-teal-600 font-medium">Explanatory</div>
              <div className="px-6 py-2 bg-purple-100 rounded-full text-purple-600 font-medium">Interactive</div>
            </div>

            <p className="text-gray-600 text-lg mb-10 max-w-2xl mx-auto">
              An advanced AI platform that combines deep learning with medical expertise to assist in
              the diagnosis and understanding of neurodegenerative diseases through MRI analysis.
            </p>
          </div>

          {/* Centered buttons with more spacing */}
          <div className="max-w-3xl mx-auto">
            <h2 className="text-2xl font-semibold text-gray-800 text-center mb-8">
              Choose Your Experience
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white bg-opacity-90 p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow flex flex-col items-center text-center">
                <div className="p-3 bg-blue-100 rounded-full mb-4">
                  <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path>
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-3">Medical Professional</h3>
                <p className="text-gray-600 mb-6 text-sm">
                  Access technical analysis with scientific terminology and detailed explainability features designed for healthcare providers.
                </p>
                <button
                  onClick={() => handleUserTypeSelect('doctor')}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-full transition-colors font-medium"
                >
                  Enter as Professional
                </button>
              </div>

              <div className="bg-white bg-opacity-90 p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow flex flex-col items-center text-center">
                <div className="p-3 bg-teal-100 rounded-full mb-4">
                  <svg className="w-8 h-8 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-3">Patient/Family</h3>
                <p className="text-gray-600 mb-6 text-sm">
                  Receive simplified explanations with accessible terminology and visual guidance designed for patients and families.
                </p>
                <button
                  onClick={() => handleUserTypeSelect('patient')}
                  className="bg-teal-500 hover:bg-teal-600 text-white px-6 py-2 rounded-full transition-colors font-medium"
                >
                  Enter as Patient
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section with transparent background - SMALLER IMAGES */}
        <div className="container mx-auto px-6 py-16 bg-transparent" id="features">
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-12">Our Advanced Capabilities</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center mb-16">
            <div className="bg-white bg-opacity-80 p-6 rounded-xl shadow-md">
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">Deep Learning Diagnostics</h3>
              <p className="text-gray-600 mb-6">
                Our platform employs state-of-the-art CNN and Vision Transformer models trained on extensive neurological datasets
                to detect subtle patterns in brain MRI scans that may indicate early signs of Alzheimer's, Parkinson's, or Multiple Sclerosis.
              </p>
              <div className="flex items-center text-blue-500">
                <span className="font-medium">Learn how it works</span>
                <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                </svg>
              </div>
            </div>
            <div className="flex justify-center">
              <img
                className="rounded-xl"
                src={dl}
                alt="Neural network visualization"
                style={{ maxHeight: "250px", objectFit: "contain" }}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
            <div className="order-2 md:order-1 flex justify-center">
              <img
                className="rounded-xl"
                src={xai}
                alt="Explainable AI visualization"
                style={{ maxHeight: "250px", objectFit: "contain" }}
              />
            </div>
            <div className="order-1 md:order-2 bg-white bg-opacity-80 p-6 rounded-xl shadow-md">
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">Explainable AI Assistant</h3>
              <p className="text-gray-600 mb-6">
                Unlike conventional AI systems, our solution provides clear visual explanations through heatmaps and attention visualizations,
                while offering personalized insights tailored specifically to your medical knowledge level and communication preferences.
              </p>
              <div className="flex items-center text-teal-500">
                <span className="font-medium">Explore the technology</span>
                <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingSection;