import React from 'react';
import icon from '../../image/icon.png';
import { Link } from "react-router-dom";
import { useNavigate } from 'react-router-dom';

function FormNav() {
  const navigate = useNavigate();
  return (
    <nav className="bg-white/0.5  fixed w-full z-20 top-20 start-60 ">
        <div className="hidden w-full md:block md:w-auto" >
         <button
                      aria-label="Back to Dashboard"
                      title="Back"
                      className="cursor-pointer p-0 border-0 outline-none bg-transparent font-bold text-indigo-600 hover:text-indigo-700"
                      onClick={() => navigate('/doctor/dashboard')}
                    >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={5.5} stroke="currentColor" className="size-8">
                     <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
                  </svg>
        </button> 
       </div>
      
    </nav>
  );
}

export default FormNav;