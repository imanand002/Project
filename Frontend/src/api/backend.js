const API_URL = "http://127.0.0.1:8000";

export async function predictEmotion(imageBlob) {
  const formData = new FormData();
  formData.append("file", imageBlob, "frame.jpg");

  const res = await fetch(`${API_URL}/predict-image`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  return data.prediction; // returns class index (0-6)
}

export const EMOTION_LABELS = [
  "Angry",
  "Disgust",
  "Fear",
  "Happy",
  "Neutral",
  "Sad",
  "Surprise"
];
