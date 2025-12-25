import React from 'react';
import icon from '../image/icon.png';
import { Link } from "react-router-dom";
function Navbar() {
  return (
    <nav className="bg-white/0.5  fixed w-full z-20 top-0 start-0 ">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">        
        <Link to="/" className="flex no-underline items-center space-x-3 rtl:space-x-reverse">
          <img src={icon} className="h-20" alt="NeuroAI Logo" />
          <span className="self-center text-2xl no-underlinetext-xl text-blue-900 font-semibold whitespace-nowrap">NeuroAI</span>
        </Link>
        <div className="hidden w-full md:block md:w-auto" id="navbar-multi-level-dropdown">
         <a href="#" className="block text-xl no-underline py-2 px-3 text-gray-700 hover:text-indigo-600">Contact</a>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;