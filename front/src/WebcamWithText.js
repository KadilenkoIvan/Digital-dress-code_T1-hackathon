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
  const downsampleCanvasRef = useRef(null); // Canvas для уменьшенного изображения
  const maskCanvasRef = useRef(null); // Canvas для маски уменьшенного размера
  const fullMaskCanvasRef = useRef(null); // Canvas для маски полного размера
  
  // Коэффициент уменьшения для модели (0.4 = 40% от оригинала)
  // Меньше значение = быстрее работа, но ниже качество
  // Рекомендуемые значения: 0.3-0.5
  const MODEL_SCALE = 0.4;
  const downsampleRatioQuality = 0.6;

  useEffect(() => {
    // Создаём временные canvas для обработки (переиспользуем на каждом кадре)
    downsampleCanvasRef.current = document.createElement('canvas');
    maskCanvasRef.current = document.createElement('canvas');
    fullMaskCanvasRef.current = document.createElement('canvas');
    
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

    // Настройка ONNX Runtime - простая конфигурация без многопоточности
    ort.env.wasm.numThreads = 1;  // Однопоточность (не требует crossOriginIsolation)
    ort.env.wasm.simd = true;     // SIMD для ускорения
    
    // Загрузка ONNX модели
    console.log("🔄 Loading model...");
    ort.InferenceSession.create("/rvm_mobilenetv3_fp32.onnx", {
      executionProviders: ['webgl', 'wasm']  // Стабильный WASM backend
    }).then((sess) => {
      console.log("✅ Model loaded successfully!");
      console.log("🎮 Backend:", "WASM (CPU with SIMD)");
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

          // Оригинальные размеры
          const origWidth = canvas.width;
          const origHeight = canvas.height;
          
          // Уменьшенные размеры для модели
          const modelWidth = Math.round(origWidth * MODEL_SCALE);
          const modelHeight = Math.round(origHeight * MODEL_SCALE);
          
          // Подготавливаем временный canvas для уменьшенного изображения
          const downsampleCanvas = downsampleCanvasRef.current;
          downsampleCanvas.width = modelWidth;
          downsampleCanvas.height = modelHeight;
          const downsampleCtx = downsampleCanvas.getContext('2d');
          
          // Рисуем уменьшенное видео
          downsampleCtx.drawImage(video, 0, 0, modelWidth, modelHeight);
          const imageData = downsampleCtx.getImageData(0, 0, modelWidth, modelHeight);
          
          const rgbData = new Float32Array(3 * modelWidth * modelHeight);
          
          // Конвертация в RGB и нормализация [0, 1]
          for (let i = 0; i < modelWidth * modelHeight; i++) {
            rgbData[i] = imageData.data[i * 4] / 255.0; // R
            rgbData[modelWidth * modelHeight + i] = imageData.data[i * 4 + 1] / 255.0; // G
            rgbData[2 * modelWidth * modelHeight + i] = imageData.data[i * 4 + 2] / 255.0; // B
          }

          // Передаём уменьшенное изображение в модель
          const inputTensor = new ort.Tensor("float32", rgbData, [1, 3, modelHeight, modelWidth]);
          // downsample_ratio - параметр внутренней оптимизации модели (0.6 = хороший баланс)
          const downsampleRatio = new ort.Tensor("float32", new Float32Array([downsampleRatioQuality]), [1]);

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
            
            // Извлечение результатов (они в уменьшенном размере)
            const phaSmall = results.pha.data;  // Маска уменьшенного размера
            
            // Обновление rec states
            if (results.r1o) recRef.current[0] = results.r1o;
            if (results.r2o) recRef.current[1] = results.r2o;
            if (results.r3o) recRef.current[2] = results.r3o;
            if (results.r4o) recRef.current[3] = results.r4o;

            // Рисуем оригинальное видео на основной canvas
            ctx.drawImage(video, 0, 0, origWidth, origHeight);
            const originalImageData = ctx.getImageData(0, 0, origWidth, origHeight);
            
            // Масштабируем маску обратно до оригинального размера
            // Используем переиспользуемые canvas
            const maskCanvas = maskCanvasRef.current;
            maskCanvas.width = modelWidth;
            maskCanvas.height = modelHeight;
            const maskCtx = maskCanvas.getContext('2d');
            const maskImageData = maskCtx.createImageData(modelWidth, modelHeight);
            
            // Заполняем уменьшенную маску (grayscale)
            for (let i = 0; i < modelWidth * modelHeight; i++) {
              const alpha = Math.min(1, Math.max(0, phaSmall[i]));
              const alphaVal = alpha * 255;
              maskImageData.data[i * 4] = alphaVal;
              maskImageData.data[i * 4 + 1] = alphaVal;
              maskImageData.data[i * 4 + 2] = alphaVal;
              maskImageData.data[i * 4 + 3] = 255;
            }
            
            maskCtx.putImageData(maskImageData, 0, 0);
            
            // Масштабируем маску до оригинального размера
            const fullMaskCanvas = fullMaskCanvasRef.current;
            fullMaskCanvas.width = origWidth;
            fullMaskCanvas.height = origHeight;
            const fullMaskCtx = fullMaskCanvas.getContext('2d');
            fullMaskCtx.drawImage(maskCanvas, 0, 0, origWidth, origHeight);
            const fullMaskData = fullMaskCtx.getImageData(0, 0, origWidth, origHeight);

            // Композитинг с фоном используя оригинальное видео и увеличенную маску
            const background = backgroundRef.current;
            const outputData = new Uint8ClampedArray(origWidth * origHeight * 4);
            
            for (let i = 0; i < origWidth * origHeight; i++) {
              const i4 = i * 4;
              const alpha = fullMaskData.data[i4] / 255.0;  // Берем альфа из увеличенной маски
              const oneMinusAlpha = 1 - alpha;
              
              // Берем цвет из оригинального видео
              const r = originalImageData.data[i4];
              const g = originalImageData.data[i4 + 1];
              const b = originalImageData.data[i4 + 2];
              
              const bgR = background ? background.data[i4] : 26;
              const bgG = background ? background.data[i4 + 1] : 26;
              const bgB = background ? background.data[i4 + 2] : 46;
              
              outputData[i4] = r * alpha + bgR * oneMinusAlpha;
              outputData[i4 + 1] = g * alpha + bgG * oneMinusAlpha;
              outputData[i4 + 2] = b * alpha + bgB * oneMinusAlpha;
              outputData[i4 + 3] = 255;
            }

            const outputImageData = new ImageData(outputData, origWidth, origHeight);
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
