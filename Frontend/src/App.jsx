// import { BrowserRouter, Routes, Route } from "react-router-dom";
// import Dashboard from "../pages/Dashboard";
// import About from "../pages/About";

// export default function App() {
//   return (
//     <BrowserRouter>
//       <Routes>
//         <Route path="/" element={<Dashboard />} />
//         <Route path="/about" element={<About />} />
//       </Routes>
//     </BrowserRouter>
//   );
// }

import Dashboard from "../src/Pages/Dashboard";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <Dashboard />
    </div>
  );
}
