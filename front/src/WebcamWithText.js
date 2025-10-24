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
  const backendNameRef = useRef('Loading...'); // Название используемого backend
  const prevMaskRef = useRef(null); // Предыдущая маска для temporal smoothing
  
  // Коэффициент уменьшения для модели (0.4 = 40% от оригинала)
  // Меньше значение = быстрее работа, но ниже качество
  // Рекомендуемые значения: 0.3-0.5
  const MODEL_SCALE = 0.3; // 0.2-0.5: меньше = быстрее работа, но ниже качество (0.25 = хороший баланс)
  const downsampleRatioQuality = 0.75; // 0.5-0.9: меньше = быстрее работа, но ниже качество (0.8 = хороший баланс)
  //MODEL_SCALE = 0.35, downsampleRatioQuality = 0.7 = хорошее качество, 45-55мс модели и 55-65мс на кадр
  //MODEL_SCALE = 0.25, downsampleRatioQuality = 0.8 = нормкальное качество (съедает наушники), 30-35мс модели и 45-55мс на кадр
  
  // Параметры постобработки маски
  const TEMPORAL_SMOOTHING = 0.95; // 0.5-0.95: больше = быстрее реакция (меньше шлейф), но больше мерцания (0.95 = хороший баланс)
  const BLUR_RADIUS = 0.35; // 0-3: радиус размытия маски (меньше = четче края, но возможны артефакты) (0.35 = хороший баланс)
  
  // Морфологические операции (Opening + Closing)
  const USE_MORPHOLOGY = false; // true/false: включить/выключить морфологические операции
  const MORPH_RADIUS = 1; // 1-2: радиус для erosion/dilation (больше = сильнее эффект, но медленнее)
  
  // Функция Erosion (сужение маски, убирает шум)
  const applyErosion = (imageData, width, height, radius) => {
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    output.set(data); // Копируем исходные данные
    
    for (let y = radius; y < height - radius; y++) {
      for (let x = radius; x < width - radius; x++) {
        let minVal = 255;
        
        // Проверяем окрестность
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const idx = ((y + dy) * width + (x + dx)) * 4;
            minVal = Math.min(minVal, data[idx]); // Берем минимум (grayscale, все каналы одинаковы)
          }
        }
        
        const idx = (y * width + x) * 4;
        output[idx] = output[idx + 1] = output[idx + 2] = minVal;
      }
    }
    
    // Копируем результат обратно
    data.set(output);
  };
  
  // Функция Dilation (расширение маски, заполняет дыры)
  const applyDilation = (imageData, width, height, radius) => {
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    output.set(data);
    
    for (let y = radius; y < height - radius; y++) {
      for (let x = radius; x < width - radius; x++) {
        let maxVal = 0;
        
        // Проверяем окрестность
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const idx = ((y + dy) * width + (x + dx)) * 4;
            maxVal = Math.max(maxVal, data[idx]); // Берем максимум
          }
        }
        
        const idx = (y * width + x) * 4;
        output[idx] = output[idx + 1] = output[idx + 2] = maxVal;
      }
    }
    
    data.set(output);
  };
  
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

    // Настройка ONNX Runtime - ПРОСТАЯ СТАБИЛЬНАЯ ВЕРСИЯ
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    
    // Загрузка модели - только WASM для стабильности
    console.log("🔄 Loading model with WASM...");
    ort.InferenceSession.create("/rvm_mobilenetv3_fp32.onnx", {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).then((sess) => {
      console.log("✅ Model loaded!");
      backendNameRef.current = 'WASM (CPU)';
      
      if (onStatsUpdate) {
        onStatsUpdate({
          fps: null,
          avgFps: null,
          modelTime: null,
          fullFrameTime: null,
          modelActive: false,
          backend: 'WASM (CPU)'
        });
      }
      
      setSession(sess);
      
      recRef.current = [
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1])
      ];
    }).catch((error) => {
      console.error("❌ Error loading model:", error);
    });

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
            
            // 1. Temporal Smoothing (EMA) - сглаживание между кадрами
            const prevMask = prevMaskRef.current;
            if (prevMask && prevMask.length === phaSmall.length) {
              for (let i = 0; i < phaSmall.length; i++) {
                phaSmall[i] = phaSmall[i] * TEMPORAL_SMOOTHING + prevMask[i] * (1 - TEMPORAL_SMOOTHING);
              }
            }
            // Сохраняем текущую маску для следующего кадра
            prevMaskRef.current = new Float32Array(phaSmall);
            
            // 2. Заполняем уменьшенную маску (grayscale)
            for (let i = 0; i < modelWidth * modelHeight; i++) {
              const alpha = Math.min(1, Math.max(0, phaSmall[i]));
              const alphaVal = alpha * 255;
              maskImageData.data[i * 4] = alphaVal;
              maskImageData.data[i * 4 + 1] = alphaVal;
              maskImageData.data[i * 4 + 2] = alphaVal;
              maskImageData.data[i * 4 + 3] = 255;
            }
            
            maskCtx.putImageData(maskImageData, 0, 0);
            
            // 3. Применяем морфологические операции (Opening + Closing)
            if (USE_MORPHOLOGY && MORPH_RADIUS > 0) {
              const morphImageData = maskCtx.getImageData(0, 0, modelWidth, modelHeight);
              
              // Opening: Erosion → Dilation (убирает шум на фоне)
              applyErosion(morphImageData, modelWidth, modelHeight, MORPH_RADIUS);
              applyDilation(morphImageData, modelWidth, modelHeight, MORPH_RADIUS);
              
              // Closing: Dilation → Erosion (заполняет дыры внутри объекта)
              applyDilation(morphImageData, modelWidth, modelHeight, MORPH_RADIUS);
              applyErosion(morphImageData, modelWidth, modelHeight, MORPH_RADIUS);
              
              maskCtx.putImageData(morphImageData, 0, 0);
            }
            
            // 4. Применяем blur на маленькой маске (быстрее чем на большой)
            if (BLUR_RADIUS > 0) {
              maskCtx.filter = `blur(${BLUR_RADIUS}px)`;
              maskCtx.drawImage(maskCanvas, 0, 0);
              maskCtx.filter = 'none';
            }
            
            // 5. Масштабируем маску до оригинального размера с билинейной интерполяцией
            const fullMaskCanvas = fullMaskCanvasRef.current;
            fullMaskCanvas.width = origWidth;
            fullMaskCanvas.height = origHeight;
            const fullMaskCtx = fullMaskCanvas.getContext('2d');
            
            // Включаем билинейную интерполяцию для плавных краев
            fullMaskCtx.imageSmoothingEnabled = true;
            fullMaskCtx.imageSmoothingQuality = 'high';
            
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
                modelActive: true,
                backend: backendNameRef.current
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
                modelActive: false,
                backend: backendNameRef.current
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
              modelActive: false,
              backend: backendNameRef.current
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
