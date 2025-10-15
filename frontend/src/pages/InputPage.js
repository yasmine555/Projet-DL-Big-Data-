import React from 'react';
import PatientInfoForm from '../components/PatientInfoForm';
import Footer from '../components/Footer';
import Navbar from '../components/Navbar';


function InputPage() {
  return (
    <div className="min-h-screen flex flex-col">

      <Navbar />
      <div className="flex-grow bg-gradient-to-b from-blue-50 to-indigo-50 pt-24 pb-12">
        <PatientInfoForm />
      </div>
      
      <Footer />
    </div>
  );
}

export default InputPage;