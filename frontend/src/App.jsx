
import './App.css'
import HomePage from './assets/pages/HomePage.jsx';
import AdminDashboard from './assets/admin/adminDashbord.jsx';
import { Routes, Route } from 'react-router-dom';
import DoctorSignIn from './assets/pages/SigninPage.jsx';
import DoctorSignUpForm from './assets/pages/SignupPage.jsx';
import ForgotPasswordRequest from './assets/pages/ForgotPasswordRequest.jsx';
import NewPassword from './assets/pages/NewPassword.jsx';
import AdminLogin from './assets/admin/AdminLoginPage.jsx';
import DoctorPage from './assets/pages/DoctorPage.jsx';
import PatientFormPage from './assets/pages/PatientFormPage.jsx';
import ResultPage from './assets/pages/ResultPage.jsx';
import ChatbotPage from './assets/pages/chatbotPage.jsx';



function App() {

  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/admin/dashboard" element={<AdminDashboard />} />
      <Route path="/admin/login" element={<AdminLogin />} />
      <Route path="/doctor/signin" element={<DoctorSignIn />} />
      <Route path="/doctor/signup" element={<DoctorSignUpForm />} />
      <Route path="/doctor/forgot-password" element={<ForgotPasswordRequest />} />
      <Route path="/doctor/reset-password" element={<NewPassword />} />
      <Route path="/doctor/dashboard" element={<DoctorPage />} />
      <Route path="/doctor/patient/form" element={<PatientFormPage />} />
      <Route path="/doctor/patient/result/:patientId?" element={<ResultPage />} />
      <Route path="/doctor/patient/result/chatbot/:patientId?" element={<ChatbotPage />} />
    </Routes>

  );
}

export default App
