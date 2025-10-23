import React, { useRef, useEffect, useState } from "react";
import * as ort from "onnxruntime-web";

function WebcamViewer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [session, setSession] = useState(null);

  useEffect(() => {
    // Получение доступа к вебкамере
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { videoRef.current.srcObject = stream; })
      .catch(err => console.error(err));

    // Подгрузка ONNX модели
    //ort.InferenceSession.create("/model.onnx").then(s => setSession(s));
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current || !session) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Преобразование в формат ONNX (mock, замените на preprocess вашей модели)
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const input = new Float32Array(imageData.data.length); // заменить на нормализацию

      // Mock: просто копируем данные (для демонстрации)
      for (let i = 0; i < imageData.data.length; i++) {
        input[i] = imageData.data[i] / 255;
      }

      // Формируем тензор и инференс
      // const tensor = new ort.Tensor("float32", input, [1, 3, canvas.height, canvas.width]);
      // const feeds = { input: tensor };
      // const output = await session.run(feeds);

      // Результат можно вывести на canvas
      // Сейчас просто оставляем оригинальное изображение
    }, 100);

    return () => clearInterval(interval);
  }, [session]);

  return (
    <div className="webcam-container">
      <video ref={videoRef} autoPlay muted />
      <canvas ref={canvasRef} />
    </div>
  );
}

export default WebcamViewer;
