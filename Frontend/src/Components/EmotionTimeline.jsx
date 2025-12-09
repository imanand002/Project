const emotionEmojis = {
  Angry: "😠",
  Disgust: "🤢",
  Fear:  "😨",
  Happy:  "😊",
  Neutral: "😐",
  Sad: "😢",
  Surprise: "😮",
};

const emotionColors = {
  Angry: "bg-red-500 hover:bg-red-600",
  Disgust: "bg-green-500 hover:bg-green-600",
  Fear:  "bg-purple-500 hover:bg-purple-600",
  Happy: "bg-yellow-400 hover:bg-yellow-500",
  Neutral: "bg-gray-500 hover: bg-gray-600",
  Sad: "bg-blue-500 hover:bg-blue-600",
  Surprise: "bg-pink-500 hover: bg-pink-600",
};

export default function EmotionTimeline({ recent }) {
  return (
    <div className="p-6 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-2xl border border-purple-500/30 backdrop-blur-sm">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <span>⏱️</span> Recent Emotions (Last 10)
      </h3>

      {recent.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {recent.map((emotion, index) => (
            <div
              key={index}
              className={`w-10 h-10 rounded-full flex items-center justify-center text-xl shadow-lg transition-all duration-200 cursor-pointer transform hover:scale-110 ${emotionColors[emotion] || "bg-gray-400"}`}
              title={emotion}
            >
              {emotionEmojis[emotion] || "❓"}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-slate-400 text-center py-4">
          No emotions detected yet. Start detecting to see history!
        </p>
      )}

      {recent.length > 0 && (
        <div className="mt-4 pt-4 border-t border-purple-500/30">
          <p className="text-xs text-slate-400">
            Latest:  <span className="text-purple-300 font-semibold">{recent[recent.length - 1]}</span>
          </p>
        </div>
      )}
    </div>
  );
}