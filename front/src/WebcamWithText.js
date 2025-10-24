import React, { useRef, useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import DraggableText from "./DraggableText";

export default function WebcamWithText({ blocks, setBlocks, selectedBlockId, setSelectedBlockId, onStatsUpdate }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [session, setSession] = useState(null);
  const recRef = useRef([]);
  const backgroundRef = useRef(null);
  const frameCountRef = useRef(0);
  const totalTimeRef = useRef(0);
  const lastStatsUpdateRef = useRef(0); // Для throttling обновления статистики

  useEffect(() => {
    // Получаем поток с камеры
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(console.error);
        }
      })
      .catch(console.error);

    // Настройка ONNX Runtime для правильной работы WebGL
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    // Загрузка ONNX модели (fp32 версия - быстрее без конвертации типов)
    ort.InferenceSession.create("/rvm_mobilenetv3_fp32.onnx", {
      executionProviders: ['wasm']  // Используем wasm (CPU с оптимизацией)
    }).then((sess) => {
      console.log("✅ Model loaded");
      console.log("🎮 Execution provider:", sess.handler?._backend?.name || "wasm");
      console.log("📊 Input names:", sess.inputNames);
      setSession(sess);
      // Инициализация recurrent states - используем float32
      recRef.current = [
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1])
      ];
    }).catch(console.error);

    // Создание фона
    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = 1280;
    bgCanvas.height = 960;
    const bgCtx = bgCanvas.getContext('2d');
    const gradient = bgCtx.createLinearGradient(0, 0, 1280, 960);
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#16213e');
    bgCtx.fillStyle = gradient;
    bgCtx.fillRect(0, 0, 1280, 960);
    backgroundRef.current = bgCtx.getImageData(0, 0, 1280, 960);
  }, []);

  useEffect(() => {
    let animationId;

    const drawFrame = async () => {
      if (!videoRef.current || !canvasRef.current) {
        animationId = requestAnimationFrame(drawFrame);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      // Проверяем, что видео уже имеет размеры
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        if (session && recRef.current.length > 0) {
          const startTime = performance.now();

          // Предобработка кадра
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          
          const width = canvas.width;
          const height = canvas.height;
          const rgbData = new Float32Array(3 * width * height);
          
          // Конвертация в RGB и нормализация [0, 1]
          for (let i = 0; i < width * height; i++) {
            rgbData[i] = imageData.data[i * 4] / 255.0; // R
            rgbData[width * height + i] = imageData.data[i * 4 + 1] / 255.0; // G
            rgbData[2 * width * height + i] = imageData.data[i * 4 + 2] / 255.0; // B
          }

          // Используем float32 - ONNX Runtime конвертирует сам на GPU
          const inputTensor = new ort.Tensor("float32", rgbData, [1, 3, height, width]);
          const downsampleRatio = new ort.Tensor("float32", new Float32Array([0.25]), [1]);

          try {
            // Запуск модели
            const feeds = {
              src: inputTensor,
              r1i: recRef.current[0],
              r2i: recRef.current[1],
              r3i: recRef.current[2],
              r4i: recRef.current[3],
              downsample_ratio: downsampleRatio
            };

            // Измерение времени только для модели
            const modelStartTime = performance.now();
            const results = await session.run(feeds);
            const modelInferenceTime = performance.now() - modelStartTime;
            
            // Извлечение результатов
            const fgr = results.fgr.data;
            const pha = results.pha.data;
            
            // Обновление rec states
            if (results.r1o) recRef.current[0] = results.r1o;
            if (results.r2o) recRef.current[1] = results.r2o;
            if (results.r3o) recRef.current[2] = results.r3o;
            if (results.r4o) recRef.current[3] = results.r4o;

            // Композитинг с фоном (оптимизированный)
            const background = backgroundRef.current;
            const outputData = new Uint8ClampedArray(width * height * 4);
            const pixelCount = width * height;
            const channelSize = pixelCount;
            
            for (let i = 0; i < pixelCount; i++) {
              const alpha = pha[i];
              const oneMinusAlpha = 1 - alpha;
              const i4 = i * 4;
              
              // Uint8ClampedArray автоматически клампит значения в [0, 255]
              const r = fgr[i] * 255;
              const g = fgr[channelSize + i] * 255;
              const b = fgr[2 * channelSize + i] * 255;
              
              const bgR = background ? background.data[i4] : 26;
              const bgG = background ? background.data[i4 + 1] : 26;
              const bgB = background ? background.data[i4 + 2] : 46;
              
              outputData[i4] = r * alpha + bgR * oneMinusAlpha;
              outputData[i4 + 1] = g * alpha + bgG * oneMinusAlpha;
              outputData[i4 + 2] = b * alpha + bgB * oneMinusAlpha;
              outputData[i4 + 3] = 255;
            }

            const outputImageData = new ImageData(outputData, width, height);
            ctx.putImageData(outputImageData, 0, 0);

            // Расчет FPS (общее время кадра)
            const frameTime = performance.now() - startTime;
            totalTimeRef.current += frameTime;
            frameCountRef.current += 1;
            const fps = 1000.0 / frameTime;
            const avgFps = (frameCountRef.current * 1000.0) / totalTimeRef.current;

            // Отправка статистики в родительский компонент (throttled - раз в 100ms)
            const now = performance.now();
            if (onStatsUpdate && (now - lastStatsUpdateRef.current) > 100) {
              lastStatsUpdateRef.current = now;
              onStatsUpdate({
                fps: fps.toFixed(2),
                avgFps: avgFps.toFixed(2),
                modelTime: modelInferenceTime.toFixed(2), // Время только модели
                fullFrameTime: frameTime.toFixed(2), // Полное время обработки кадра
                modelActive: true
              });
            }
          } catch (error) {
            console.error("Model inference error:", error);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const now = performance.now();
            if (onStatsUpdate && (now - lastStatsUpdateRef.current) > 100) {
              lastStatsUpdateRef.current = now;
              onStatsUpdate({
                fps: null,
                avgFps: null,
                modelTime: null,
                fullFrameTime: null,
                modelActive: false
              });
            }
          }
        } else {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const now = performance.now();
          if (onStatsUpdate && (now - lastStatsUpdateRef.current) > 100) {
            lastStatsUpdateRef.current = now;
            onStatsUpdate({
              fps: null,
              avgFps: null,
              modelTime: null,
              fullFrameTime: null,
              modelActive: false
            });
          }
        }
      }

      animationId = requestAnimationFrame(drawFrame);
    };

    animationId = requestAnimationFrame(drawFrame);

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [session]);

  const handleUpdate = (id, newProps) => {
    setBlocks((prev) => prev.map((b) => (b.id === id ? { ...b, ...newProps } : b)));
  };

  const handleBackgroundClick = () => setSelectedBlockId(null);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "1280px",
        height: "960px",
        overflow: "hidden",
      }}
      onClick={handleBackgroundClick}
    >
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{ display: "none" }}
      />
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", background: "black" }}
      />

      {blocks.map((b) => (
        <DraggableText
          key={b.id}
          block={b}
          selected={b.id === selectedBlockId}
          onSelect={setSelectedBlockId}
          onUpdate={handleUpdate}
          parentRef={containerRef}
        />
      ))}
    </div>
  );
}
