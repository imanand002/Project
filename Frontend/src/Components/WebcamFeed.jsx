import { useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { predictEmotion, EMOTION_LABELS } from "../api/backend";

export default function WebcamFeed({ onPrediction }) {
  const webcamRef = useRef(null);

  // Capture frame every 1 second
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!webcamRef.current) return;

      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      const blob = await (await fetch(screenshot)).blob();
      const predictionIndex = await predictEmotion(blob);

      onPrediction(EMOTION_LABELS[predictionIndex]);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center">
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="rounded-xl shadow-lg"
      />
    </div>
  );
}
