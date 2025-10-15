import React from 'react';
import LandingSection from '../components/LandingSection';
import Footer from '../components/Footer';
import Navbar from '../components/Navbar';

function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-grow">
        <LandingSection />
      </main>
      <Footer />
    </div>
  );
}

export default HomePage;