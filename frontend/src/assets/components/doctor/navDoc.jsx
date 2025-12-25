import React from 'react';
import { Link } from 'react-router-dom';
import icon from '../../image/icon.png';


function NavDoc({ onSignOut }) {
  return (

    <nav className="bg-white/0.5  fixed w-full z-20 top-10 start-40 ">
          <div className="hidden w-full md:block md:w-auto" >

          <button
            type="button"
            onClick={onSignOut}
            className="cursor-pointer text-white bg-red-600 hover:bg-red-700 focus:ring-4 focus:ring-red-300 shadow-xs font-medium leading-5 rounded-full text-sm px-4 py-2.5 focus:outline-none"
          >
            Sign out
          </button>  </div>
    </nav>


  );
}

export default NavDoc;