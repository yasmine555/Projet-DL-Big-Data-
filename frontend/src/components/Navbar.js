import Ract from 'react';
function Navbar() {
    return(
          <nav className="fixed top-0 left-0 right-0  z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-xl">
                <a href="/">N</a>
              </div>
              <a href="/" className="ml-3 text-xl font-semibold text-gray-800">NeuroAI</a>
            </div>
            
          </div>
        </div>
      </nav>)
}

export default Navbar;
