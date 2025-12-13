import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import FormNav from '../components/patient-form/navPatient';
import { createPatient, updatePatient, askQuery, getPatientMRIScans, analyzePatientMRI } from '../../services/api';

import BasicInfoSection from '../components/patient-form/BasicInfoSection';
import SymptomsSection from '../components/patient-form/SymptomsSection';
import LifestyleSection from '../components/patient-form/LifestyleSection';
import MedicalHistorySection from '../components/patient-form/MedicalHistorySection';
import TestsSection from '../components/patient-form/TestsSection';
import MRIUploadSection from '../components/patient-form/MRIUploadSection';
import Toast from '../components/ui/Toast.jsx';

export default function PatientFormPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const [userType, setUserType] = useState('patient');
  const [mode, setMode] = useState('create');
  const [patientId, setPatientId] = useState(null);
  const [fromDashboard, setFromDashboard] = useState(false);

  const [formData, setFormData] = useState({
    patientName: '',
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
    neuroExamNotes: '',
    mmse: '',
    moca: '',
    amyloid: 'unknown',
    tau: 'unknown',
    apoe4: 'unknown',
  });

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mriScans, setMriScans] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [activeSection, setActiveSection] = useState('basic');
  const [toast, setToast] = useState({ show: false, message: '', type: 'success' });

  useEffect(() => {
    const state = location.state || {};
    const storedUserType = localStorage.getItem('userType') || 'patient';
    const currentUserType = state.userType || storedUserType;
    setUserType(currentUserType);
    setFromDashboard(!!state.fromDashboard);

    if (state.mode === 'edit' && state.patient) {
      setMode('edit');
      setPatientId(state.patient.id);
      const p = state.patient;
      setFormData(prev => ({
        ...prev,
        patientName: p.name || '',
        patientAge: p.age || '',
        patientGender: p.sex || '',
        educationLevel: p.education_level || '',
        familyHistory: p.family_history === 'Yes',
        cognitiveSymptoms: p.symptoms_list || [],
        motorSymptoms: p.metrics?.motorSymptoms || [],
        sleepIssues: !!p.metrics?.sleepIssues,
        physicalActivity: p.metrics?.lifestyle?.physicalActivity || '',
        smokingStatus: p.metrics?.lifestyle?.smokingStatus || '',
        alcoholConsumption: p.metrics?.lifestyle?.alcoholConsumption || '',
        existingConditions: (typeof p.medical_history === 'string' ? p.medical_history.split(',').map(s => s.trim()) : p.medical_history) || [],
        currentMedications: '',
        neuroExamNotes: p.neuro_exam_notes || '',
        mmse: p.mmse_score || p.mse || '',
        moca: p.moca_score || p.moca || '',
        amyloid: p.biomarkers?.amyloid || 'unknown',
        tau: p.biomarkers?.tau || 'unknown',
        apoe4: p.biomarkers?.apoe4 || 'unknown',
      }));
      // Load latest MRI scan preview if available
      if (p.id) {
        (async () => {
          try {
            const scans = await getPatientMRIScans(p.id);
            setMriScans(scans?.scans || []);
            const imageUrl = scans?.scans?.[0]?.image_url;
            if (imageUrl) setPreviewUrl(imageUrl);
          } catch (err) {
            console.warn('Failed to load MRI scans', err);
          }
        })();
      }
      }
    }, [location, userType]);

  const sections = [
    { id: 'basic', label: 'Basic Info' },
    { id: 'symptoms', label: 'Symptoms' },
    { id: 'lifestyle', label: 'Lifestyle' },
    { id: 'medical', label: 'Medical History' },
    ...(userType === 'doctor' ? [{ id: 'tests', label: 'Clinical Tests' }] : []),
    { id: 'mri', label: 'MRI Upload' },
  ];

  const validateForm = () => {
    const newErrors = {};
    if (userType === 'doctor' && !formData.patientName) newErrors.patientName = 'Name is required';
    if (!formData.patientAge) newErrors.patientAge = 'Age is required';
    if (!formData.patientGender) newErrors.patientGender = 'Gender is required';
    if (mode === 'create' && !selectedFile) newErrors.file = 'MRI scan is required';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    if (!validateForm()) return;
    setIsLoading(true);
    try {
      const patientData = {
        name: formData.patientName || `Patient ${Date.now()}`,
        age: parseInt(formData.patientAge),
        sex: formData.patientGender,
        education_level: formData.educationLevel,
        family_history: formData.familyHistory ? 'Yes' : 'No',
        medical_history: formData.existingConditions.join(', '),
        symptoms_list: [...formData.cognitiveSymptoms, ...formData.motorSymptoms],
        neuro_exam_notes: formData.neuroExamNotes,
        mse: parseFloat(formData.mmse) || null,
        moca: parseFloat(formData.moca) || null,
        metrics: {
          biomarkers: {
            amyloid: formData.amyloid,
            tau: formData.tau,
            apoe4: formData.apoe4,
          },
          lifestyle: {
            physicalActivity: formData.physicalActivity,
            smokingStatus: formData.smokingStatus,
            alcoholConsumption: formData.alcoholConsumption,
          },
          sleepIssues: !!formData.sleepIssues,
          motorSymptoms: formData.motorSymptoms,
        },
      };

      let savedPatient;
      if (userType === 'doctor') {
        if (mode === 'edit' && patientId) {
          savedPatient = await updatePatient(patientId, patientData);
        } else {
          savedPatient = await createPatient(patientData);
        }
      } else {
        savedPatient = { ...patientData, id: 'temp_' + Date.now() };
      }

      const metadata = {
        ...patientData,
        medical_history: formData.existingConditions,
        mmse_score: parseFloat(formData.mmse) || null,
        moca_score: parseFloat(formData.moca) || null,
        biomarkers: { amyloid: formData.amyloid, tau: formData.tau, apoe4: formData.apoe4 },
        user_type: userType,
      };

      // If doctor and an MRI file is selected, upload and analyze it now
      try {
        if (userType === 'doctor' && selectedFile && savedPatient?.id) {
          const mriRes = await analyzePatientMRI(savedPatient.id, selectedFile);
          if (mriRes?.image_url) {
            setPreviewUrl(mriRes.image_url);
            metadata.imaging_findings = mriRes.prediction_class;
            metadata.mri_uploaded = true;
            // refresh scans list
            try {
              const scans = await getPatientMRIScans(savedPatient.id);
              setMriScans(scans?.scans || []);
            } catch (e) {}
          }
        }
      } catch (mriErr) {
        console.warn('MRI upload failed:', mriErr);
        setToast({ show: true, type: 'error', message: `MRI upload failed: ${mriErr?.message || 'Unexpected error'}` });
      }

      const responses = [0];
      const analysisResult = await askQuery(responses, metadata, savedPatient.id);

      sessionStorage.setItem('patient_id', savedPatient.id);
      sessionStorage.setItem('patient_context', JSON.stringify(metadata));
      sessionStorage.setItem('questionnaire_result', JSON.stringify(analysisResult));
      if (previewUrl) sessionStorage.setItem('mri_preview', previewUrl);
      setToast({ show: true, type: 'success', message: 'Form submitted successfully. Redirecting to AI Analysis...' });
      navigate('/results');
    } catch (error) {
      console.error('Error submitting data:', error);
      setToast({ show: true, type: 'danger', message: 'Error submitting data. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  // Navigation between tabs is handled by the tab buttons above.

  return (
    <div className="min-h-screen bg-neutral-100">
      
      {userType === 'doctor' && (
        <FormNav />   )}
      <div className="p-20">
        <div className="container mx-auto px-8 py-10 max-w-3xl ">
        

          <div className="text-center mb-10">
            <h1 className=" font-bold text-gray-800 mb-4">
              {mode === 'edit' ? 'Edit Patient Record' : (userType === 'doctor' ? 'Clinical Assessment Form' : 'Health Information Form')}
            </h1>
            <div className="flex flex-wrap justify-center gap-4 mb-6">
              <div className={`px-6 py-2 rounded-full  font-medium ${userType === 'doctor' ? 'bg-indigo-100 text-indigo-600' : 'bg-teal-100 text-teal-600'}`}>
                {userType === 'doctor' ? 'Professional Mode' : 'Patient Mode'}
              </div>
            </div>
          </div>

          <div className="mb-8 overflow-x-auto">
            <div className="flex min-w-max justify-center space-x-2 pb-2">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`px-6  py-3 rounded-lg font-medium transition-colors ${activeSection === section.id
                    ? (userType === 'doctor' ? 'bg-indigo-600 text-white' : 'bg-teal-600 text-white')
                    : 'bg-white/80 hover:bg-white text-gray-700'}`}
                >
                  {section.label}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="bg-white/90  shadow-xl rounded-xl overflow-hidden backdrop-blur-sm">
              <BasicInfoSection
                visible={activeSection === 'basic'}
                userType={userType}
                formData={formData}
                errors={errors}
                onChange={setFormData}
              />
              <SymptomsSection
                visible={activeSection === 'symptoms'}
                formData={formData}
                onChange={setFormData}
              />
              <LifestyleSection
                visible={activeSection === 'lifestyle'}
                formData={formData}
                onChange={setFormData}
              />
              <MedicalHistorySection
                visible={activeSection === 'medical'}
                userType={userType}
                formData={formData}
                onChange={setFormData}
              />
              {userType === 'doctor' && (
                <TestsSection
                  visible={activeSection === 'tests'}
                  formData={formData}
                  onChange={setFormData}
                />
              )}
              <MRIUploadSection
                visible={activeSection === 'mri'}
                errors={errors}
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onFileChange={setSelectedFile}
                onPreviewChange={setPreviewUrl}
                scans={mriScans}
              />

              
            </div>
            {activeSection === 'mri' && (
              <div className="p-5 bg-gradient-to-b from-transparent to-indigo-50 flex justify-end">
                <button type="submit" disabled={isLoading} className={`btn-primary btn-sm flex items-center ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}>
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
              </div>
            )}
          </form>
          {toast.show && (
            <Toast type={toast.type} message={toast.message} onClose={() => setToast(prev => ({ ...prev, show: false }))} />
          )}
        </div>
      </div>
    </div>
  );
}