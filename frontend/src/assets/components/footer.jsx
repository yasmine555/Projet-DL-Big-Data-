import React from 'react';

function Footer() {
  return (
    <footer className="footer sm:footer-horizontal z-20 top-0 start-0 footer-center text-base-content p-4">
      <aside>
        <p>Copyright © {new Date().getFullYear()} - All rights reserved</p>
      </aside>
    </footer>
  );
}

export default Footer;