export default function PredictionCard({ emotion }) {
  return (
    <div className="p-6 bg-white rounded-xl shadow-lg text-center">
      <h2 className="text-2xl font-bold">Current Emotion</h2>
      <p className="text-4xl mt-3 text-blue-600 font-semibold">
        {emotion ? emotion : "Detecting..."}
      </p>
    </div>
  );
}
