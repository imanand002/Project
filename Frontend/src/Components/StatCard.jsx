export default function StatCard({ label, value, icon, color = "purple" }) {
  const colorClasses = {
    purple: "from-purple-600 to-purple-700",
    pink: "from-pink-600 to-pink-700",
    blue: "from-blue-600 to-blue-700",
    green: "from-green-600 to-green-700",
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-xl p-6 shadow-lg hover:shadow-2xl transition-all transform hover:scale-105`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-white/70 text-sm font-medium">{label}</p>
          <p className="text-white text-3xl font-black mt-2">{value}</p>
        </div>
        <span className="text-4xl opacity-20">{icon}</span>
      </div>
    </div>
  );
}