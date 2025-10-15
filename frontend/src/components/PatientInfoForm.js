import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function PatientInfoForm() {
  const navigate = useNavigate();
  const [userType, setUserType] = useState('patient');
  const [formData, setFormData] = useState({
    patientAge: '',
    patientGender: '',
    educationLevel: '',
    familyHistory: false,
    cognitiveSymptoms: [],
    motorSymptoms: [],
    sleepIssues: false,
    physicalActivity: '',
    smokingStatus: '',
    alcoholConsumption: '',
    existingConditions: [],
    currentMedications: '',
  });
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [activeSection, setActiveSection] = useState('basic');

  useEffect(() => {
    // Get user type from localStorage
    const storedUserType = localStorage.getItem('userType') || 'patient';
    setUserType(storedUserType);
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setFormData({
      ...formData,
      [name]: checked
    });
  };

  const handleMultiSelectChange = (e, category) => {
    const value = e.target.value;
    if (formData[category].includes(value)) {
      setFormData({
        ...formData,
        [category]: formData[category].filter(item => item !== value)
      });
    } else {
      setFormData({
        ...formData,
        [category]: [...formData[category], value]
      });
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create a preview URL for the image
      const fileReader = new FileReader();
      fileReader.onload = () => {
        setPreviewUrl(fileReader.result);
      };
      fileReader.readAsDataURL(file);
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.patientAge) newErrors.patientAge = "Age is required";
    if (!formData.patientGender) newErrors.patientGender = "Gender is required";
    if (!selectedFile) newErrors.file = "MRI scan is required";
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    try {
      // Create FormData object to send file and form data
      const submitData = new FormData();
      submitData.append('mriScan', selectedFile);
      submitData.append('patientData', JSON.stringify(formData));
      submitData.append('userType', userType);
      
      // Here you would make an API call to your backend
      // const response = await apiService.submitPatientData(submitData);
      
      // For now, simulate API call with timeout
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Navigate to results page
      navigate('/results');
    } catch (error) {
      console.error('Error submitting data:', error);
      alert('There was an error submitting your data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const sections = [
    { id: 'basic', label: 'Basic Info' },
    { id: 'symptoms', label: 'Symptoms' },
    { id: 'lifestyle', label: 'Lifestyle' },
    { id: 'medical', label: 'Medical History' },
    { id: 'mri', label: 'MRI Upload' },
  ];

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          {userType === 'doctor' ? 'Clinical Assessment Form' : 'Health Information Form'}
        </h1>
        
        <div className="flex flex-wrap justify-center gap-4 mb-6">
          <div className={`px-6 py-2 rounded-full text-lg font-medium ${userType === 'doctor' ? 'bg-blue-100 text-blue-600' : 'bg-teal-100 text-teal-600'}`}>
            {userType === 'doctor' ? 'Professional Mode' : 'Patient Mode'}
          </div>
        </div>
        
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          {userType === 'doctor' 
            ? "Please provide the following clinical information to enhance our AI's diagnostic accuracy."
            : "Please fill out this form to help us better understand your health situation."}
        </p>
      </div>

      {/* Section Navigation */}
      <div className="mb-8 overflow-x-auto">
        <div className="flex min-w-max justify-center space-x-2 pb-2">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`px-6 py-3 rounded-lg text-lg font-medium transition-colors ${
                activeSection === section.id
                  ? userType === 'doctor' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-teal-500 text-white'
                  : 'bg-white bg-opacity-70 hover:bg-opacity-100 text-gray-600'
              }`}
            >
              {section.label}
            </button>
          ))}
        </div>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="bg-white bg-opacity-90 shadow-xl rounded-2xl overflow-hidden backdrop-blur-sm">
          {/* Basic Information */}
          <div className={`p-8 ${activeSection !== 'basic' && 'hidden'}`}>
            <div className="flex items-center mb-8 border-b pb-4">
              <div className="p-3 rounded-full mr-4 bg-gradient-to-r from-blue-100 to-indigo-100">
                <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800">Basic Information</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="patientAge">
                  Age
                </label>
                <input 
                  type="number" 
                  id="patientAge" 
                  name="patientAge" 
                  value={formData.patientAge} 
                  onChange={handleInputChange} 
                  className={`shadow-md appearance-none border-2 ${errors.patientAge ? 'border-red-300' : 'border-blue-100'} rounded-xl w-full py-3 px-4 text-lg text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-transparent transition duration-200`}
                  min="0"
                  max="120"
                  placeholder="Enter age"
                />
                {errors.patientAge && <p className="text-red-500 text-sm mt-1">{errors.patientAge}</p>}
              </div>
              
              <div>
                <label className="block text-gray-700 text-lg font-medium mb-3">
                  Gender
                </label>
                <div className="grid grid-cols-3 gap-4">
                  <label className="bg-white border-2 border-blue-100 rounded-xl p-4 flex items-center justify-center hover:bg-blue-50 transition cursor-pointer">
                    <input 
                      type="radio" 
                      name="patientGender" 
                      value="male" 
                      checked={formData.patientGender === 'male'} 
                      onChange={handleInputChange} 
                      className="form-radio h-5 w-5 text-blue-500 mr-2"
                    />
                    <span className="text-lg text-gray-700">Male</span>
                  </label>
                  <label className="bg-white border-2 border-blue-100 rounded-xl p-4 flex items-center justify-center hover:bg-blue-50 transition cursor-pointer">
                    <input 
                      type="radio" 
                      name="patientGender" 
                      value="female" 
                      checked={formData.patientGender === 'female'} 
                      onChange={handleInputChange} 
                      className="form-radio h-5 w-5 text-blue-500 mr-2"
                    />
                    <span className="text-lg text-gray-700">Female</span>
                  </label>
                  
                </div>
                {errors.patientGender && <p className="text-red-500 text-sm mt-1">{errors.patientGender}</p>}
             </div>
              
              <div>
                <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="educationLevel">
                  Education Level
                </label>
                <select 
                  id="educationLevel" 
                  name="educationLevel" 
                  value={formData.educationLevel} 
                  onChange={handleInputChange} 
                  className="shadow-md appearance-none border-2 border-blue-100 rounded-xl w-full py-3 px-4 text-lg text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-transparent transition duration-200"
                >
                  <option value="">Select education level</option>
                  <option value="lessThanHighSchool">Less than high school</option>
                  <option value="highSchool">High school</option>
                  <option value="someCollege">Some college</option>
                  <option value="bachelor">Bachelor's degree</option>
                  <option value="graduate">Graduate degree</option>
                </select>
              </div>
              
              <div className="flex items-center">
                <label className="bg-white border-2 border-blue-100 rounded-xl p-4 flex items-center hover:bg-blue-50 transition cursor-pointer w-full">
                  <input 
                    type="checkbox" 
                    name="familyHistory" 
                    checked={formData.familyHistory} 
                    onChange={handleCheckboxChange} 
                    className="form-checkbox h-5 w-5 text-blue-500 mr-3"
                  />
                  <span className="text-lg text-gray-700">Family history of neurodegenerative diseases</span>
                </label>
              </div>
            </div>
          </div>
          
          {/* Symptoms */}
          <div className={`p-8 bg-gradient-to-br from-blue-50 to-indigo-50 ${activeSection !== 'symptoms' && 'hidden'}`}>
            <div className="flex items-center mb-8 border-b pb-4">
              <div className="p-3 rounded-full mr-4 bg-gradient-to-r from-purple-100 to-indigo-100">
                <svg className="w-8 h-8 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800">Symptoms & Health</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <label className="block text-gray-700 text-lg font-medium mb-3">
                  Cognitive Symptoms
                </label>
                <div className="bg-white rounded-xl p-5 shadow-md border border-purple-100">
                  <div className="grid grid-cols-1 gap-3">
                    {[
                      {value: 'memoryLoss', label: 'Memory loss'},
                      {value: 'difficultyConcentrating', label: 'Difficulty concentrating'},
                      {value: 'confusionDisorientation', label: 'Confusion/disorientation'},
                      {value: 'difficultyPlanning', label: 'Difficulty planning or problem solving'},
                      {value: 'languageProblems', label: 'Language or speech problems'},
                      {value: 'visualPerceptionIssues', label: 'Visual perception issues'}
                    ].map(option => (
                      <label key={option.value} className="inline-flex items-center bg-purple-50 hover:bg-purple-100 transition p-3 rounded-lg cursor-pointer">
                        <input 
                          type="checkbox" 
                          value={option.value} 
                          checked={formData.cognitiveSymptoms.includes(option.value)}
                          onChange={(e) => handleMultiSelectChange(e, 'cognitiveSymptoms')}
                          className="form-checkbox h-5 w-5 text-purple-500 mr-3"
                        />
                        <span className="text-gray-700 text-lg">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              
              <div>
                <label className="block text-gray-700 text-lg font-medium mb-3">
                  Motor Symptoms
                </label>
                <div className="bg-white rounded-xl p-5 shadow-md border border-purple-100">
                  <div className="grid grid-cols-1 gap-3">
                    {[
                      {value: 'tremors', label: 'Tremors'},
                      {value: 'slowMovement', label: 'Slow movement'},
                      {value: 'rigidity', label: 'Muscle rigidity'},
                      {value: 'balanceProblems', label: 'Balance problems'},
                      {value: 'coordinationIssues', label: 'Coordination issues'},
                      {value: 'difficultyWalking', label: 'Difficulty walking'}
                    ].map(option => (
                      <label key={option.value} className="inline-flex items-center bg-purple-50 hover:bg-purple-100 transition p-3 rounded-lg cursor-pointer">
                        <input 
                          type="checkbox" 
                          value={option.value} 
                          checked={formData.motorSymptoms.includes(option.value)}
                          onChange={(e) => handleMultiSelectChange(e, 'motorSymptoms')}
                          className="form-checkbox h-5 w-5 text-purple-500 mr-3"
                        />
                        <span className="text-gray-700 text-lg">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="md:col-span-2">
                <label className="bg-white rounded-xl p-5 shadow-md border border-purple-100 flex items-center hover:bg-purple-50 transition cursor-pointer w-full">
                  <input 
                    type="checkbox" 
                    name="sleepIssues" 
                    checked={formData.sleepIssues} 
                    onChange={handleCheckboxChange} 
                    className="form-checkbox h-5 w-5 text-purple-500 mr-3"
                  />
                  <span className="text-lg text-gray-700">Do you experience sleep disturbances (insomnia, excessive daytime sleepiness, etc.)?</span>
                </label>
              </div>
            </div>
          </div>
          
          {/* Lifestyle */}
          <div className={`p-8 ${activeSection !== 'lifestyle' && 'hidden'}`}>
            <div className="flex items-center mb-8 border-b pb-4">
              <div className="p-3 rounded-full mr-4 bg-gradient-to-r from-teal-100 to-green-100">
                <svg className="w-8 h-8 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800">Lifestyle Factors</h2>
            </div>
            
            <div className="grid grid-cols-1 gap-8">
              <div>
                <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="physicalActivity">
                  Physical Activity Level
                </label>
                <div className="bg-gradient-to-r from-teal-50 to-green-50 rounded-xl p-3">
                  <select 
                    id="physicalActivity" 
                    name="physicalActivity" 
                    value={formData.physicalActivity} 
                    onChange={handleInputChange} 
                    className="shadow-md appearance-none border-2 border-teal-100 rounded-xl w-full py-3 px-4 text-lg text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-teal-300 focus:border-transparent transition duration-200"
                  >
                    <option value="">Select activity level</option>
                    <option value="sedentary">Sedentary (little to no exercise)</option>
                    <option value="light">Light activity (1-3 days/week)</option>
                    <option value="moderate">Moderate activity (3-5 days/week)</option>
                    <option value="active">Very active (6-7 days/week)</option>
                    <option value="veryActive">Extra active (professional athlete level)</option>
                  </select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="smokingStatus">
                    Smoking Status
                  </label>
                  <div className="bg-gradient-to-r from-teal-50 to-green-50 rounded-xl p-3">
                    <select 
                      id="smokingStatus" 
                      name="smokingStatus" 
                      value={formData.smokingStatus} 
                      onChange={handleInputChange} 
                      className="shadow-md appearance-none border-2 border-teal-100 rounded-xl w-full py-3 px-4 text-lg text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-teal-300 focus:border-transparent transition duration-200"
                    >
                      <option value="">Select smoking status</option>
                      <option value="never">Never smoked</option>
                      <option value="former">Former smoker</option>
                      <option value="occasional">Occasional smoker</option>
                      <option value="regular">Regular smoker</option>
                    </select>
                  </div>
                </div>
                
                <div>
                  <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="alcoholConsumption">
                    Alcohol Consumption
                  </label>
                  <div className="bg-gradient-to-r from-teal-50 to-green-50 rounded-xl p-3">
                    <select 
                      id="alcoholConsumption" 
                      name="alcoholConsumption" 
                      value={formData.alcoholConsumption} 
                      onChange={handleInputChange} 
                      className="shadow-md appearance-none border-2 border-teal-100 rounded-xl w-full py-3 px-4 text-lg text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-teal-300 focus:border-transparent transition duration-200"
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
          </div>
          
          {/* Medical History */}
          <div className={`p-8 bg-gradient-to-br from-blue-50 to-indigo-50 ${activeSection !== 'medical' && 'hidden'}`}>
            <div className="flex items-center mb-8 border-b pb-4">
              <div className="p-3 rounded-full mr-4 bg-gradient-to-r from-pink-100 to-red-100">
                <svg className="w-8 h-8 text-pink-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.5 12.75l6 6 9-13.5"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800">Medical History</h2>
            </div>
            
            <div>
              <label className="block text-gray-700 text-lg font-medium mb-3">
                Existing Conditions
              </label>
              <div className="bg-white rounded-xl p-5 shadow-md border border-pink-100">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    {value: 'hypertension', label: 'Hypertension'},
                    {value: 'diabetes', label: 'Diabetes'},
                    {value: 'heartDisease', label: 'Heart Disease'},
                    {value: 'stroke', label: 'Previous Stroke'},
                    {value: 'tbi', label: 'Traumatic Brain Injury'},
                    {value: 'depression', label: 'Depression/Anxiety'},
                    {value: 'sleepApnea', label: 'Sleep Apnea'},
                    {value: 'thyroidDisorder', label: 'Thyroid Disorder'}
                  ].map(option => (
                    <label key={option.value} className="inline-flex items-center bg-pink-50 hover:bg-pink-100 transition p-3 rounded-lg cursor-pointer">
                      <input 
                        type="checkbox" 
                        value={option.value} 
                        checked={formData.existingConditions.includes(option.value)}
                        onChange={(e) => handleMultiSelectChange(e, 'existingConditions')}
                        className="form-checkbox h-5 w-5 text-pink-500 mr-3"
                      />
                      <span className="text-gray-700 text-lg">{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              
              {userType === 'doctor' && (
                <div className="mt-8">
                  <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="currentMedications">
                    Current Medications (comma separated)
                  </label>
                  <textarea 
                    id="currentMedications" 
                    name="currentMedications" 
                    value={formData.currentMedications} 
                    onChange={handleInputChange} 
                    className="shadow-md appearance-none border-2 border-pink-100 rounded-xl w-full py-3 px-4 text-lg text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-pink-300 focus:border-transparent transition duration-200"
                    rows="3"
                    placeholder="E.g., Levodopa, Memantine, Rivastigmine, etc."
                  ></textarea>
                </div>
              )}
            </div>
          </div>
          
          {/* MRI Upload */}
          <div className={`p-8 ${activeSection !== 'mri' && 'hidden'}`}>
            <div className="flex items-center mb-8 border-b pb-4">
              <div className="p-3 rounded-full mr-4 bg-gradient-to-r from-indigo-100 to-blue-100">
                <svg className="w-8 h-8 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800">Brain MRI Upload</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              <div>
               
                
                <div className="mb-6">
                  <label className="block text-gray-700 text-lg font-medium mb-3" htmlFor="mriUpload">
                    Upload MRI Scan
                  </label>
                  <div className={`border-2 border-dashed rounded-xl p-6 text-center ${errors.file ? 'border-red-300 bg-red-50' : 'border-blue-200 bg-blue-50'} hover:bg-blue-100 transition cursor-pointer`}>
                    <input 
                      type="file" 
                      id="mriUpload" 
                      accept=".dcm,.nii,.nii.gz,.jpg,.jpeg,.png" 
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <label htmlFor="mriUpload" className="cursor-pointer flex flex-col items-center">
                      <svg className="w-16 h-16 text-blue-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                      </svg>
                      <span className="text-lg font-medium text-blue-600 mb-2">
                        Click to upload MRI scan
                      </span>
                     
                    </label>
                  </div>
                  {errors.file && <p className="text-red-500 text-sm mt-1">{errors.file}</p>}
                </div>
              </div>
              
              <div className="flex justify-center">
                {previewUrl ? (
                  <div className="border-4 border-blue-200 rounded-xl overflow-hidden shadow-lg h-64 w-full">
                    <img 
                      src={previewUrl} 
                      alt="MRI Preview" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="border-4 border-blue-100 rounded-xl p-8 h-64 w-full flex flex-col items-center justify-center bg-blue-50">
                    <svg className="w-16 h-16 text-blue-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                    </svg>
                    <p className="text-lg text-blue-400 text-center">
                      MRI preview will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Navigation and Submit */}
          <div className="p-8 bg-gradient-to-b from-transparent to-blue-50 flex flex-col md:flex-row justify-between items-center">
            <div className="flex space-x-4 mb-4 md:mb-0">
              {activeSection !== sections[0].id && (
                <button 
                  type="button"
                  onClick={() => {
                    const currentIndex = sections.findIndex(s => s.id === activeSection);
                    setActiveSection(sections[currentIndex - 1].id);
                  }}
                  className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-6 rounded-full flex items-center transition"
                >
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
                  </svg>
                  Previous
                </button>
              )}
              
              {activeSection !== sections[sections.length - 1].id && (
                <button 
                  type="button"
                  onClick={() => {
                    const currentIndex = sections.findIndex(s => s.id === activeSection);
                    setActiveSection(sections[currentIndex + 1].id);
                  }}
                  className={`${userType === 'doctor' ? 'bg-blue-500 hover:bg-blue-600' : 'bg-teal-500 hover:bg-teal-600'} text-white font-medium py-2 px-6 rounded-full flex items-center transition`}
                >
                  Next
                  <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                  </svg>
                </button>
              )}
            </div>
            
            {activeSection === 'mri' && (
              <button 
                type="submit" 
                className={`${userType === 'doctor' ? 'bg-blue-500 hover:bg-blue-600' : 'bg-teal-500 hover:bg-teal-600'} text-white font-bold py-3 px-8 rounded-full focus:outline-none focus:shadow-outline transition flex items-center ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isLoading}
              >
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
                    <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7"></path>
                    </svg>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}

export default PatientInfoForm;