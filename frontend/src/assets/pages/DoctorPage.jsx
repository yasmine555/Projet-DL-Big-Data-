import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import NavDoc from '../components/doctor/navDoc';
import PatientList from '../components/doctor/PatientList';
import DoctorHeader from '../components/doctor/DoctorHeader';
import DoctorActions from '../components/doctor/DoctorActions';
import { listPatients, deletePatient, getPatientResults, getPatientById } from '../../services/api';

function DoctorPage() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [query, setQuery] = useState('');

  const load = async () => {
    try {
      const data = await listPatients();
      const mapped = (data || []).map(p => {
        const patientId = p._id || p.id || p.patient_id;
        return {
          id: patientId,
          name: p.name,
          age: p.age ?? '—',
          lastVisit: p.created_at ? new Date(p.created_at).toLocaleString() : '—',
          raw: p,
        };
      });
      setPatients(mapped);
    } catch (e) {
      console.error('Error loading patients:', e);
      setPatients([]);
    }
  };

  useEffect(() => { load(); }, []);

  const handleSignOut = () => {
    localStorage.removeItem('token');
    navigate('/doctor/signin');
  };

  const handleAddPatient = () => {
    navigate('/doctor/patient/form', { state: { userType: 'doctor', fromDashboard: true } });
  };

  const handleViewPatient = async (patientId) => {
    if (!patientId) return;
    try {
      const fullPatientData = await getPatientById(patientId);
      const context = {
        user_type: 'doctor',
        age: fullPatientData.age ?? null,
        sex: fullPatientData.sex ?? null,
        education_level: fullPatientData.education_level ?? null,
        years_of_education: fullPatientData.years_of_education ?? null,
        mmse_score: fullPatientData.mmse_score ?? fullPatientData.mse ?? null,
        moca_score: fullPatientData.moca_score ?? fullPatientData.moca ?? null,
        family_history: fullPatientData.family_history ?? false,
        symptoms_list: fullPatientData.symptoms_list || [],
        biomarkers: fullPatientData.biomarkers || { amyloid: 'unknown', tau: 'unknown', apoe4: 'unknown' },
        medical_history: fullPatientData.medical_history || [],
        imaging_findings: fullPatientData.imaging_findings,
        neuro_exam_notes: fullPatientData.neuro_exam_notes
      };
      sessionStorage.setItem('patient_id', patientId);
      sessionStorage.setItem('patient_context', JSON.stringify(context));
      sessionStorage.setItem('patient_profile', JSON.stringify(fullPatientData));
      try {
        const results = await getPatientResults(patientId);
        if (results?.[0]) {
          sessionStorage.setItem('questionnaire_result', JSON.stringify(results[0]));
        } else {
          sessionStorage.setItem('questionnaire_result', JSON.stringify({
            short_summary: 'No AI analysis has been performed for this patient yet.',
            answer: 'Please update the patient record or submit a new assessment to generate an AI analysis.',
            patient_summary: 'Patient profile loaded successfully.',
            sources: [],
            reasoning_trace: []
          }));
        }
      } catch {
        sessionStorage.setItem('questionnaire_result', JSON.stringify({
          short_summary: 'No AI analysis has been performed for this patient yet.',
          answer: 'Please update the patient record or submit a new assessment to generate an AI analysis.',
          patient_summary: 'Patient profile loaded successfully.',
          sources: [],
          reasoning_trace: []
        }));
      }
      navigate(`/doctor/patient/result/${patientId}`);
      sessionStorage.setItem('currentPatientId', patientId);
    } catch (error) {
      console.error('Error loading patient:', error);
      alert('Failed to load patient data. Please try again.');
    }
  };

  const handleEditPatient = async (patientId) => {
    if (!patientId) return;
    try {
      const fullPatientData = await getPatientById(patientId);
      navigate('/doctor/patient/form', {
        state: { userType: 'doctor', patient: fullPatientData, mode: 'edit', fromDashboard: true }
      });
    } catch (error) {
      console.error('Error loading patient for edit:', error);
      alert('Failed to load patient data for editing. Please try again.');
    }
  };

  const handleDeletePatient = async (patientId) => {
    try {
      await deletePatient(patientId);
      await load();
    } catch (error) {
      console.error('Error deleting patient:', error);
      alert('Failed to delete patient.');
    }
  };

  return (

    <div className="min-h-screen bg-neutral-100">

      <NavDoc onSignOut={() => { localStorage.removeItem('token'); navigate('/doctor/signin') }} />
      <div className=" pt-20 pb-20">
        <div className="container mx-auto px-8 py-10">
          <div className="flex items-center justify-between mb-8">

            <DoctorHeader />
            <DoctorActions onAddPatient={handleAddPatient} />
          </div>
          <div className="mb-6 max-w-md">
            <label htmlFor="patient-search" className="block text-sm font-medium text-gray-700 mb-1">Search Patient</label>
            <div className="relative">
              <input
                id="patient-search"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by name or ID..."
                className="input w-50 pl-10 ml-10"
              />
              <span className="absolute left-3 top-1/2  -translate-y-1/2 text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
                  <path fillRule="evenodd" d="M10.5 3.75a6.75 6.75 0 104.31 12.02l4.21 4.22a.75.75 0 101.06-1.06l-4.22-4.21A6.75 6.75 0 0010.5 3.75zm0 1.5a5.25 5.25 0 100 10.5 5.25 5.25 0 000-10.5z" clipRule="evenodd" />
                </svg>
              </span>
            </div>
          </div>
          <PatientList
            patients={patients.filter(p => {
              const q = query.trim().toLowerCase();
              if (!q) return true;
              const name = (p.name || '').toLowerCase();
              const idStr = String(p.id || '').toLowerCase();
              return name.includes(q) || idStr.includes(q);
            })}
            onView={handleViewPatient}
            onEdit={handleEditPatient}
            onDelete={handleDeletePatient}
          />
        </div>
      </div>
    </div>
  );
}

export default DoctorPage;

