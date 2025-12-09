import { useState } from "react";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-gray-900 text-white px-6 py-4 shadow-md fixed w-full top-0 z-50">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        
        {/* Logo */}
        <h1 className="text-2xl font-bold">EmotionSense</h1>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-8">
          <a href="/" className="hover:text-blue-400 transition">Dashboard</a>
          <a href="/about" className="hover:text-blue-400 transition">About</a>
        </div>

        {/* Mobile Menu Button */}
        <button 
          className="md:hidden" 
          onClick={() => setMenuOpen(!menuOpen)}
        >
          <span className="text-3xl">&#9776;</span>
        </button>
      </div>

      {/* Mobile Dropdown */}
      {menuOpen && (
        <div className="md:hidden mt-3 space-y-2 px-4 pb-4">
          <a href="/" className="block hover:text-blue-400">Dashboard</a>
          <a href="/about" className="block hover:text-blue-400">About</a>
        </div>
      )}
    </nav>
  );
}
