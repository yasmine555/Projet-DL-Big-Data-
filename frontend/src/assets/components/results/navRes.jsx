import React from 'react';
import icon from '../../image/icon.png';
import { Link } from "react-router-dom";
import { useNavigate } from 'react-router-dom';
import { useParams } from 'react-router-dom';


function ResNav() {
  const navigate = useNavigate();
  const { patientId } = useParams();
  const userType = localStorage.getItem('userType') || 'patient';

  const resultLink = userType === 'doctor'
    ? `/doctor/patient/result/${patientId}`
    : `/patient/result/${patientId}`;

  const chatLink = userType === 'doctor'
    ? `/doctor/patient/result/chatbot/${patientId}`
    : `/patient/result/chatbot/${patientId}`;

  return (
    <nav className="  fixed h-16 w-full z-20 top-0 start-0">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
        <Link to="/" className="flex no-underline items-center space-x-3 rtl:space-x-reverse">
          <img src={icon} className="h-16" alt="NeuroAI Logo"></img>
          <span className="self-center text-2xl text-blue-900 font-semibold whitespace-nowrap">NeuroAI</span>
        </Link>
        <div className="items-center justify-between hidden w-full md:flex md:w-auto md:order" id="navbar-sticky">
          <ul className="flex flex-col p-4 md:p-0 mt-4 list-none font-medium border border-gray-200 rounded-lg bg-gray-50 md:space-x-8 rtl:space-x-reverse md:flex-row md:mt-0 md:border-0 md:bg-transparent">

            {userType === 'doctor' && (
              <li>
                <button
                  onClick={() => navigate("/doctor/dashboard")}
                  className="block py-2 px-3 text-white bg-indigo-600 rounded-lg md:bg-transparent md:text-indigo-600 md:p-0 hover:text-indigo-700 no-underline"
                >
                  Back to Dashboard
                </button>
              </li>
            )}

            <li>
              <Link to={resultLink} className="block py-2 px-3 text-gray-700 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-indigo-600 md:p-0 no-underline">Results</Link>
            </li>

            <li>
              <Link to={chatLink} className="block py-2 px-3 text-gray-700 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-indigo-600 md:p-0 no-underline">AI Assistance</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>

  );
}

export default ResNav;