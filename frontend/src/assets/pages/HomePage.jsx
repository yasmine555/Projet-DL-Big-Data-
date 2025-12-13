import React from 'react';
import LandingSection from '../components/landingPage';
import Navbar from '../components/navBar';

function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-grow">
        <LandingSection />
      </main>
      
    </div>
  );
}

export default HomePage;