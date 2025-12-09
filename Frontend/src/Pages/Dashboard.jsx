// import Navbar from "./components/Navbar";
// import WebcamFeed from "./components/WebcamFeed";
// import PredictionCard from "./components/PredictionCard";
// import ChartSection from "./components/ChartSection";

// export default function Dashboard() {
//   return (
//     <div className="min-h-screen bg-gray-100">
//       <Navbar />

//       <div className="pt-24 max-w-6xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-10">
        
//         {/* Webcam + Prediction */}
//         <div className="space-y-6">
//           <WebcamFeed />
//           <PredictionCard />
//         </div>

//         {/* Charts */}
//         <div>
//           <ChartSection />
//         </div>

//       </div>
//     </div>
//   );
// }

// import { useState } from "react";
// import WebcamFeed from "src/Components/WebcamFeed";
// import PredictionCard from "src/Components/PredictionCard";
// import ChartSection from "src/Components/ChartSection";

import { useState } from "react";
import WebcamFeed from "../Components/WebcamFeed";
import PredictionCard from "../Components/PredictionCard";
import ChartSection from "../Components/ChartSection";


export default function Dashboard() {
  const [emotion, setEmotion] = useState(null);

  const [history, setHistory] = useState({
    Angry: 0,
    Disgust: 0,
    Fear: 0,
    Happy: 0,
    Neutral: 0,
    Sad: 0,
    Surprise: 0
  });

  function handlePrediction(newEmotion) {
    setEmotion(newEmotion);
    setHistory(prev => ({
      ...prev,
      [newEmotion]: prev[newEmotion] + 1
    }));
  }

  return (
    <div className="p-6 grid grid-cols-2 gap-6">
      <WebcamFeed onPrediction={handlePrediction} />

      <PredictionCard emotion={emotion} />

      <div className="col-span-2">
        <ChartSection history={history} />
      </div>
    </div>
  );
}
