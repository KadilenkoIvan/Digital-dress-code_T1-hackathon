import React, { useRef, useEffect, useState, useMemo } from "react";
import * as ort from "onnxruntime-web";
import DraggableText from "./DraggableText";
import DraggableImage from "./DraggableImage";
import TextEditorPanel from "./TextEditorPanel";
import ImageEditorPanel from "./ImageEditorPanel";
import "./TextEditorPanel.css";

export default function WebcamWithText({ blocks, setBlocks, selectedBlockId, setSelectedBlockId, onStatsUpdate, backgroundImage, backgroundBlur = 0, modelScale = 0.4, downsampleRatio = 0.8, rawMode = false, numThreads = 1 }) {
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
  const deviceNameRef = useRef('Loading...'); // Название используемого device
  const prevMaskRef = useRef(null); // Предыдущая маска для temporal smoothing
  const lastEmployeeRef = useRef(null); // Последний обработанный employee для предотвращения дублирования
  const textClickedRef = useRef(false); // Флаг для отслеживания кликов по тексту
  const backgroundLayerRef = useRef(null); // Ref для фонового слоя
  const blurredBgCanvasRef = useRef(null); // Canvas для размытого реального фона
  const frameSkipCounter = useRef(0); // Счетчик кадров для пропуска (обрабатываем каждый второй)
  const cachedMaskRef = useRef(null); // Кэшированная маска для пропущенного кадра
  const skipFramesCount = useRef(1); // Сколько кадров пропускать (1 = обрабатываем каждый второй)
  const lastModelTimeRef = useRef(0); // Последнее время модели для отображения на пропущенных кадрах
  
  // Параметр пропуска кадров - можно управлять через UI
  const FRAMES_TO_SKIP = 1; // 1 = каждый второй, 2 = каждый третий, 0 = все обрабатывать
  
  // Коэффициент уменьшения для модели (0.4 = 40% от оригинала)
  // Меньше значение = быстрее работа, но ниже качество
  // Рекомендуемые значения: 0.1-1.0 (управляется через UI)
  // modelScale и downsampleRatio передаются как пропсы из App.js
  //Примеры производительности:
  //modelScale = 0.35, downsampleRatio = 0.7 = хорошее качество, 45-55мс модели и 55-65мс на кадр
  //modelScale = 0.25, downsampleRatio = 0.8 = нормальное качество (съедает наушники), 30-35мс модели и 45-55мс на кадр
  //modelScale = 0.2, downsampleRatio = 0.7 = так себе качество (съедает руки), 20-25мс модели и 30-35мс на кадр
  //modelScale = 0.3, downsampleRatio = 0.7 = нормальное качество, 35-40мс модели и 45-50мс на кадр
  //modelScale = 0.2, downsampleRatio = 0.8 = нормальное качество, 25мс модели и 35-40мс на кадр
  
  // Параметры предобработки входного изображения
  const USE_GAMMA_CORRECTION = false; // true/false: коррекция яркости для улучшения контраста
  const GAMMA = 1; // 1.0-1.3: гамма-коррекция (>1 = осветление темных областей, улучшает сегментацию)
  
  // Параметры постобработки маски
  const TEMPORAL_SMOOTHING = 0.85; // 0.5-0.95: больше = быстрее реакция (меньше шлейф), но больше мерцания
  const ADAPTIVE_SMOOTHING = true; // true/false: адаптивное сглаживание (меньше шлейф при движении)
  const BLUR_RADIUS = 0.35; // 0-3: радиус размытия маски (меньше = четче края, но возможны артефакты)
  
  // Параметры bilateral blur для входного изображения
  const USE_BILATERAL_BLUR = false; // true/false: включить/выключить bilateral blur
  const BILATERAL_RADIUS = 2; // 1-5: радиус размытия (больше = сильнее размытие)
  const BILATERAL_SPATIAL_SIGMA = 2.0; // 1.0-5.0: пространственная сигма (влияет на пространственное размытие)
  const BILATERAL_COLOR_SIGMA = 30.0; // 10.0-50.0: цветовая сигма (сохранение краев по цвету)
  const LIGHTING_THRESHOLD = 0.3; // 0.1-0.5: порог яркости для применения размытия (меньше = чаще применяется)
  
  // Морфологические операции (Opening + Closing) - убирают шум и заполняют дыры
  const USE_MORPHOLOGY = true; // true/false: включить/выключить морфологические операции
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
  
  // Функция Гамма-коррекции (осветляет темные области, улучшает контраст)
  const applyGammaCorrection = (imageData, gamma) => {
    const data = imageData.data;
    const gammaCorrection = 1 / gamma;
    
    // Предвычисляем таблицу для ускорения (256 значений)
    const lookupTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      lookupTable[i] = Math.min(255, Math.max(0, 255 * Math.pow(i / 255, gammaCorrection)));
    }
    
    // Применяем к каждому пикселю (только RGB, не альфа)
    for (let i = 0; i < data.length; i += 4) {
      data[i] = lookupTable[data[i]];         // R
      data[i + 1] = lookupTable[data[i + 1]]; // G
      data[i + 2] = lookupTable[data[i + 2]]; // B
    }
  };
  
  // Функция для вычисления разницы между масками (для адаптивного smoothing)
  const calculateMaskDifference = (mask1, mask2) => {
    if (!mask1 || !mask2 || mask1.length !== mask2.length) return 1.0;
    
    let totalDiff = 0;
    for (let i = 0; i < mask1.length; i++) {
      totalDiff += Math.abs(mask1[i] - mask2[i]);
    }
    return totalDiff / mask1.length;
  };
  
  // Функция bilateral blur (сохраняет края, размывает шум)
  const applyBilateralBlur = (imageData, width, height, radius, spatialSigma, colorSigma) => {
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    
    // Предвычисляем весовые функции для ускорения
    const spatialWeights = new Array(radius * 2 + 1);
    const colorWeights = new Array(256);
    
    // Пространственные веса (гауссова функция)
    for (let i = -radius; i <= radius; i++) {
      spatialWeights[i + radius] = Math.exp(-(i * i) / (2 * spatialSigma * spatialSigma));
    }
    
    // Цветовые веса
    for (let i = 0; i < 256; i++) {
      colorWeights[i] = Math.exp(-(i * i) / (2 * colorSigma * colorSigma));
    }
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerIdx = (y * width + x) * 4;
        const centerR = data[centerIdx];
        const centerG = data[centerIdx + 1];
        const centerB = data[centerIdx + 2];
        
        let weightedSumR = 0, weightedSumG = 0, weightedSumB = 0;
        let totalWeight = 0;
        
        // Проходим по окрестности
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const neighborIdx = (ny * width + nx) * 4;
              const neighborR = data[neighborIdx];
              const neighborG = data[neighborIdx + 1];
              const neighborB = data[neighborIdx + 2];
              
              // Пространственный вес
              const spatialWeight = spatialWeights[dx + radius] * spatialWeights[dy + radius];
              
              // Цветовой вес (разность цветов)
              const colorDiffR = Math.abs(centerR - neighborR);
              const colorDiffG = Math.abs(centerG - neighborG);
              const colorDiffB = Math.abs(centerB - neighborB);
              const avgColorDiff = (colorDiffR + colorDiffG + colorDiffB) / 3;
              
              const colorWeight = colorWeights[Math.round(avgColorDiff)];
              
              const weight = spatialWeight * colorWeight;
              
              weightedSumR += neighborR * weight;
              weightedSumG += neighborG * weight;
              weightedSumB += neighborB * weight;
              totalWeight += weight;
            }
          }
        }
        
        output[centerIdx] = weightedSumR / totalWeight;
        output[centerIdx + 1] = weightedSumG / totalWeight;
        output[centerIdx + 2] = weightedSumB / totalWeight;
        output[centerIdx + 3] = data[centerIdx + 3]; // Альфа-канал без изменений
      }
    }
    
    // Копируем результат обратно
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

    // Настройка ONNX Runtime - МНОГОПОТОЧНЫЙ WASM с проверкой
    const canUseMultiThread = window.crossOriginIsolated === true;
    
    // Проверяем сохранённое значение из localStorage (после перезагрузки)
    const savedThreads = localStorage.getItem('onnx_num_threads');
    const maxThreads = Math.min(navigator.hardwareConcurrency || 4, 6);
    let threadsToUse = numThreads;
    
    if (savedThreads) {
      const savedValue = parseInt(savedThreads);
      // Валидация сохранённого значения
      if (savedValue > maxThreads || savedValue < 1) {
        console.warn(`⚠️ Invalid saved threads ${savedValue}, using default: 1`);
        localStorage.setItem('onnx_num_threads', '1');
        threadsToUse = 1;
      } else {
        threadsToUse = savedValue;
      }
    }
    
    threadsToUse = canUseMultiThread ? threadsToUse : 1;
    
    console.log(`💻 CPU cores available: ${navigator.hardwareConcurrency || 4}, requested: ${numThreads}, using: ${threadsToUse}`);
    console.log(`🔗 Multi-threading available: ${canUseMultiThread ? 'YES' : 'NO (missing HTTP headers)'}`);
    console.log(`🚀 Initializing with ${threadsToUse} thread(s)`);
    
    ort.env.wasm.numThreads = threadsToUse;
    ort.env.wasm.simd = true;
    
    const deviceName = canUseMultiThread ? `WASM (${threadsToUse} threads)` : 'WASM (1 thread)';
    console.log(`🔄 Loading model with ${deviceName}...`);
    
    ort.InferenceSession.create("/rvm_mobilenetv3_fp32.onnx", {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
    }).then((sess) => {
      console.log(`✅ Model loaded!`);
      deviceNameRef.current = deviceName;
      
      if (onStatsUpdate) {
        onStatsUpdate({
          fps: null,
          avgFps: null,
          modelTime: null,
          fullFrameTime: null,
          modelActive: false,
          device: deviceName
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

  // Создание фона
  useEffect(() => {
    const updateBackground = () => {
      if (!videoRef.current) return;
      const video = videoRef.current;

      // Ждём, пока видео определит размеры
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        requestAnimationFrame(updateBackground);
        return;
      }

      const canvasW = video.videoWidth;
      const canvasH = video.videoHeight;

      const bgCanvas = document.createElement('canvas');
      bgCanvas.width = canvasW;
      bgCanvas.height = canvasH;
      const bgCtx = bgCanvas.getContext('2d');

      if (backgroundImage) {
        const img = new Image();
        img.onload = () => {
          const imgW = img.width;
          const imgH = img.height;

          // Масштабирование как в CSS background-size: cover
          const scale = Math.max(canvasW / imgW, canvasH / imgH);
          const scaledW = imgW * scale;
          const scaledH = imgH * scale;
          const offsetX = (canvasW - scaledW) / 2;
          const offsetY = (canvasH - scaledH) / 2;

          // Применяем размытие к фону для скрытия артефактов композитинга
          if (backgroundBlur > 0) {
            bgCtx.filter = `blur(${backgroundBlur}px)`;
          }
          bgCtx.drawImage(img, offsetX, offsetY, scaledW, scaledH);
          bgCtx.filter = 'none'; // Сбрасываем фильтр
          
          backgroundRef.current = bgCtx.getImageData(0, 0, canvasW, canvasH);
        };
        img.src = backgroundImage;
      } else {
        // fallback: градиент, под размер видео
        const gradient = bgCtx.createLinearGradient(0, 0, canvasW, canvasH);
        gradient.addColorStop(0, '#1a1a2e');
        gradient.addColorStop(1, '#16213e');
        bgCtx.fillStyle = gradient;
        bgCtx.fillRect(0, 0, canvasW, canvasH);
        backgroundRef.current = bgCtx.getImageData(0, 0, canvasW, canvasH);
      }
    };
  updateBackground();
}, [backgroundImage, backgroundBlur]);

  // Сброс lastEmployeeRef при изменении фона
  useEffect(() => {
    // Сбрасываем lastEmployeeRef, чтобы текстовые блоки могли быть пересозданы
    lastEmployeeRef.current = null;
    console.log("🔄 Background changed, lastEmployeeRef reset");
  }, [backgroundImage]);

  // Управление canvas реального видео при загрузке/удалении фона
  useEffect(() => {
    if (!backgroundLayerRef.current) return;
    
    if (backgroundImage) {
      // Если загружен фон, удаляем canvas реального видео
      if (blurredBgCanvasRef.current && blurredBgCanvasRef.current.parentNode) {
        backgroundLayerRef.current.removeChild(blurredBgCanvasRef.current);
        blurredBgCanvasRef.current = null;
        console.log("🗑️ Real video background canvas removed (background loaded)");
      }
      // Устанавливаем загруженное изображение
      backgroundLayerRef.current.style.backgroundImage = `url(${backgroundImage})`;
    } else {
      // Если фон удален, убираем CSS фон (canvas создастся автоматически в drawFrame)
      backgroundLayerRef.current.style.backgroundImage = 'none';
    }
  }, [backgroundImage]);

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

        // Если включен режим "Сырое видео", просто выводим камеру без обработки
        if (rawMode) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Обновляем статистику (throttled - раз в 100ms, как в обработанном режиме)
          const now = performance.now();
          if (onStatsUpdate && (now - lastStatsUpdateRef.current) > 100) {
            lastStatsUpdateRef.current = now;
            onStatsUpdate({
              fps: null,
              avgFps: null,
              modelTime: null,
              fullFrameTime: null,
              modelActive: false,
              device: 'RAW MODE'
            });
          }
          
          animationId = requestAnimationFrame(drawFrame);
          return;
        }

        if (session && recRef.current.length > 0) {
          const startTime = performance.now();

          // Оригинальные размеры
          const origWidth = canvas.width;
          const origHeight = canvas.height;
          
          // Уменьшенные размеры для модели (используется значение из UI)
          const modelWidth = Math.round(origWidth * modelScale);
          const modelHeight = Math.round(origHeight * modelScale);
          
          // Подготавливаем временный canvas для уменьшенного изображения
          const downsampleCanvas = downsampleCanvasRef.current;
          downsampleCanvas.width = modelWidth;
          downsampleCanvas.height = modelHeight;
          const downsampleCtx = downsampleCanvas.getContext('2d');
          
          // Рисуем уменьшенное видео
          downsampleCtx.drawImage(video, 0, 0, modelWidth, modelHeight);
          const imageData = downsampleCtx.getImageData(0, 0, modelWidth, modelHeight);
          
          // Применяем гамма-коррекцию для улучшения контраста (особенно в темных условиях)
          if (USE_GAMMA_CORRECTION && GAMMA !== 1.0) {
            applyGammaCorrection(imageData, GAMMA);
          }
          
          // Применяем bilateral blur всегда
          if (USE_BILATERAL_BLUR) {
            applyBilateralBlur(imageData, modelWidth, modelHeight, BILATERAL_RADIUS, BILATERAL_SPATIAL_SIGMA, BILATERAL_COLOR_SIGMA);
          }
          
          const rgbData = new Float32Array(3 * modelWidth * modelHeight);
          
          // Конвертация в RGB и нормализация [0, 1]
          for (let i = 0; i < modelWidth * modelHeight; i++) {
            rgbData[i] = imageData.data[i * 4] / 255.0; // R
            rgbData[modelWidth * modelHeight + i] = imageData.data[i * 4 + 1] / 255.0; // G
            rgbData[2 * modelWidth * modelHeight + i] = imageData.data[i * 4 + 2] / 255.0; // B
          }

          // Передаём уменьшенное изображение в модель
          const inputTensor = new ort.Tensor("float32", rgbData, [1, 3, modelHeight, modelWidth]);
          // downsample_ratio - параметр внутренней оптимизации модели
          const downsampleRatioTensor = new ort.Tensor("float32", new Float32Array([downsampleRatio]), [1]);

          // Пропуск кадров: обрабатываем каждый (FRAMES_TO_SKIP+1) кадр
          frameSkipCounter.current = (frameSkipCounter.current + 1) % (FRAMES_TO_SKIP + 1);
          const shouldRunModel = FRAMES_TO_SKIP === 0 ? true : frameSkipCounter.current === FRAMES_TO_SKIP;

          try {
            
            let phaSmall;
            let modelInferenceTime = 0;
            
            if (shouldRunModel) {
              // Запуск модели
              const feeds = {
                src: inputTensor,
                r1i: recRef.current[0],
                r2i: recRef.current[1],
                r3i: recRef.current[2],
                r4i: recRef.current[3],
                downsample_ratio: downsampleRatioTensor
              };

              // Измерение времени только для модели
              const modelStartTime = performance.now();
              const results = await session.run(feeds);
              modelInferenceTime = performance.now() - modelStartTime;
              
              // Извлечение результатов (они в уменьшенном размере)
              phaSmall = results.pha.data;  // Маска уменьшенного размера
              
              // Сохраняем маску для следующего кадра
              cachedMaskRef.current = new Float32Array(phaSmall);
              
              // Сохраняем время модели для отображения на пропущенных кадрах
              lastModelTimeRef.current = modelInferenceTime;
              
              // Обновление rec states
              if (results.r1o) recRef.current[0] = results.r1o;
              if (results.r2o) recRef.current[1] = results.r2o;
              if (results.r3o) recRef.current[2] = results.r3o;
              if (results.r4o) recRef.current[3] = results.r4o;
            } else {
              // Используем кэшированную маску и последнее время модели
              phaSmall = cachedMaskRef.current;
              modelInferenceTime = lastModelTimeRef.current;
              
              if (!phaSmall) {
                // Если кэша нет (первый пропущенный кадр), используем текущий кадр
                const feeds = {
                  src: inputTensor,
                  r1i: recRef.current[0],
                  r2i: recRef.current[1],
                  r3i: recRef.current[2],
                  r4i: recRef.current[3],
                  downsample_ratio: downsampleRatioTensor
                };
                
                // Измеряем время для первого кадра
                const modelStartTime = performance.now();
                const results = await session.run(feeds);
                modelInferenceTime = performance.now() - modelStartTime;
                lastModelTimeRef.current = modelInferenceTime;
                
                phaSmall = results.pha.data;
                cachedMaskRef.current = new Float32Array(phaSmall);
                
                // Обновление rec states
                if (results.r1o) recRef.current[0] = results.r1o;
                if (results.r2o) recRef.current[1] = results.r2o;
                if (results.r3o) recRef.current[2] = results.r3o;
                if (results.r4o) recRef.current[3] = results.r4o;
              }
            }

            // Очищаем canvas (делаем полностью прозрачным)
            ctx.clearRect(0, 0, origWidth, origHeight);
            
            // Рисуем оригинальное видео на временный canvas
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
            // ЗАКОМЕНТИРОВАНО для ускорения
            /*
            const prevMask = prevMaskRef.current;
            let smoothingCoeff = TEMPORAL_SMOOTHING;
            
            if (prevMask && prevMask.length === phaSmall.length) {
              // Адаптивное сглаживание: при большом движении используем меньший коэффициент
              if (ADAPTIVE_SMOOTHING) {
                const diff = calculateMaskDifference(phaSmall, prevMask);
                // Если изменение большое (движение), используем больший вес новой маски
                // diff в диапазоне [0, 1], обычно 0.01-0.3
                // При diff > 0.1 (сильное движение) увеличиваем реактивность
                if (diff > 0.05) {
                  smoothingCoeff = Math.max(0.3, TEMPORAL_SMOOTHING - diff * 2);
                }
              }
              
              for (let i = 0; i < phaSmall.length; i++) {
                phaSmall[i] = phaSmall[i] * smoothingCoeff + prevMask[i] * (1 - smoothingCoeff);
              }
            }
            // Сохраняем текущую маску для следующего кадра
            prevMaskRef.current = new Float32Array(phaSmall);
            */
            
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

            // Если фон не загружен - используем реальное видео в качестве фона (с размытием или без)
            if (!backgroundImage) {
              if (!blurredBgCanvasRef.current) {
                blurredBgCanvasRef.current = document.createElement('canvas');
                blurredBgCanvasRef.current.style.position = 'absolute';
                blurredBgCanvasRef.current.style.top = '0';
                blurredBgCanvasRef.current.style.left = '0';
                blurredBgCanvasRef.current.style.width = '100%';
                blurredBgCanvasRef.current.style.height = '100%';
                blurredBgCanvasRef.current.style.objectFit = 'cover';
                
                // Вставляем canvas в фоновый слой
                if (backgroundLayerRef.current) {
                  // Убираем CSS фон
                  backgroundLayerRef.current.style.backgroundImage = 'none';
                  // Добавляем canvas
                  backgroundLayerRef.current.appendChild(blurredBgCanvasRef.current);
                  console.log("🎥 Real video background canvas created");
                }
              }
              
              const blurredBgCanvas = blurredBgCanvasRef.current;
              
              // Устанавливаем размеры только если они изменились (чтобы не сбрасывать контекст)
              if (blurredBgCanvas.width !== origWidth || blurredBgCanvas.height !== origHeight) {
                blurredBgCanvas.width = origWidth;
                blurredBgCanvas.height = origHeight;
              }
              
              const blurredBgCtx = blurredBgCanvas.getContext('2d');
              
              // Очищаем canvas перед рисованием нового кадра
              blurredBgCtx.clearRect(0, 0, origWidth, origHeight);
              
              // Рисуем оригинальное видео с размытием (если backgroundBlur > 0) или без
              if (backgroundBlur > 0) {
                blurredBgCtx.filter = `blur(${backgroundBlur}px)`;
              }
              blurredBgCtx.drawImage(video, 0, 0, origWidth, origHeight);
              blurredBgCtx.filter = 'none';
            }

            // Композитинг: выводим только человека с прозрачным фоном
            // Canvas будет прозрачным, чтобы через него было видно текст и фон
            const outputData = new Uint8ClampedArray(origWidth * origHeight * 4);
            
            for (let i = 0; i < origWidth * origHeight; i++) {
              const i4 = i * 4;
              const alpha = fullMaskData.data[i4] / 255.0;  // Берем альфа из увеличенной маски
              
              // Берем цвет из оригинального видео
              const r = originalImageData.data[i4];
              const g = originalImageData.data[i4 + 1];
              const b = originalImageData.data[i4 + 2];
              
              // Выводим только человека, фон делаем прозрачным
              outputData[i4] = r;
              outputData[i4 + 1] = g;
              outputData[i4 + 2] = b;
              outputData[i4 + 3] = alpha * 255; // Альфа-канал: 255 для человека, 0 для фона
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
                modelTime: modelInferenceTime.toFixed(2), // Время только модели (показываем время последнего вызова)
                fullFrameTime: `${frameTime.toFixed(2)} (skip: ${FRAMES_TO_SKIP})`, // Полное время обработки кадра + параметр пропуска
                modelActive: true,
                device: deviceNameRef.current
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
                device: deviceNameRef.current
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
              device: deviceNameRef.current
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
  }, [session, modelScale, downsampleRatio, rawMode, backgroundImage, backgroundBlur, numThreads]);

  // Перезагрузка модели при изменении numThreads
  useEffect(() => {
    if (!session) return; // Модель ещё не загружена
    
    console.log(`🔄 numThreads changed to ${numThreads}, forcing full reload...`);
    
    const canUseMultiThread = window.crossOriginIsolated === true;
    const threadsToUse = canUseMultiThread ? numThreads : 1;
    
    if (!canUseMultiThread && numThreads > 1) {
      console.warn(`⚠️ Multi-threading unavailable! Restart dev server to enable. Using 1 thread.`);
    }
    
    // Закрываем старую сессию
    if (session) {
      session.release?.().catch(console.error);
    }
    setSession(null);
    
    // КРИТИЧНО: Нужно полностью перезагрузить страницу для смены количества потоков
    // WASM модуль компилируется один раз и не может быть изменён динамически
    console.warn('⚠️ ONNX Runtime cannot dynamically change thread count.');
    console.warn('💡 Please RELOAD THE PAGE (F5) to apply new thread count.');
    console.warn(`🔄 Page will reload in 1 second to apply ${threadsToUse} threads...`);
    
    // Сохраняем новое значение потоков в localStorage
    localStorage.setItem('onnx_num_threads', threadsToUse.toString());
    
    // Перезагружаем страницу через 1 секунду
    setTimeout(() => {
      window.location.reload();
    }, 1000);
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numThreads]);
  
  // Сброс рекуррентных состояний при изменении modelScale или downsampleRatio
  useEffect(() => {
    if (recRef.current.length > 0) {
      // Сбрасываем рекуррентные состояния к начальным значениям
      recRef.current = [
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
        new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1])
      ];
      console.log("🔄 Recurrent states reset due to parameter change. modelScale:", modelScale, "downsampleRatio:", downsampleRatio);
    }
  }, [modelScale, downsampleRatio]);

  // Сброс счетчиков статистики при переключении режимов или изменении параметров
  useEffect(() => {
    frameCountRef.current = 0;
    totalTimeRef.current = 0;
    lastStatsUpdateRef.current = 0;
    console.log("📊 Stats counters reset. rawMode:", rawMode);
    
    // Явное обновление статистики при переключении режима
    if (onStatsUpdate) {
      if (rawMode) {
        onStatsUpdate({
          fps: null,
          avgFps: null,
          modelTime: null,
          fullFrameTime: null,
          modelActive: false,
          device: 'RAW MODE'
        });
      } else {
        onStatsUpdate({
          fps: null,
          avgFps: null,
          modelTime: null,
          fullFrameTime: null,
          modelActive: false,
          device: deviceNameRef.current
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawMode, modelScale, downsampleRatio, numThreads]);

  // Мемоизируем блок "b1" и его ключевые свойства для оптимизации
  const bgBlockData = useMemo(() => {
    const bgBlock = blocks.find(b => b.id === "b1");
    if (!bgBlock?.employee) return null;
    return {
      employee: bgBlock.employee,
      level: bgBlock.level || bgBlock.employee.privacy_level || "low"
    };
  }, [blocks]);

  useEffect(() => {
    if (!bgBlockData) {
      lastEmployeeRef.current = null;
      return;
    }

    const { employee, level: privacyLevel } = bgBlockData;
    const bgBlock = blocks.find(b => b.id === "b1");

    // Проверяем, не обрабатывали ли мы уже этого employee
    const logoUrl = employee.branding?.logo_url || '';
    const employeeKey = `${employee.full_name}_${employee.position}_${privacyLevel}_${logoUrl}`;
    if (lastEmployeeRef.current === employeeKey) {
      return;
    }
    
    // Сохраняем текущего employee как последнего обработанного
    lastEmployeeRef.current = employeeKey;

    const textBlocks = [];

    // Функция для расчета "обратного" цвета к фону
    const getInverseColor = (x, y, width, height) => {
      const bg = backgroundRef.current;
      if (!bg) return "white";
      const data = bg.data;
      let rSum = 0, gSum = 0, bSum = 0, count = 0;

      for (let dy = 0; dy < height; dy++) {
        for (let dx = 0; dx < width; dx++) {
          const px = x + dx;
          const py = y + dy;
          if (px >= bg.width || py >= bg.height) continue;
          const idx = (py * bg.width + px) * 4;
          rSum += data[idx];
          gSum += data[idx + 1];
          bSum += data[idx + 2];
          count++;
        }
      }

      const rAvg = rSum / count;
      const gAvg = gSum / count;
      const bAvg = bSum / count;

      // Инвертируем цвета
      const rInv = 255 - rAvg;
      const gInv = 255 - gAvg;
      const bInv = 255 - bAvg;

      return `rgb(${Math.round(rInv)}, ${Math.round(gInv)}, ${Math.round(bInv)})`;
    };

    // Генерация текста с динамическим цветом
    const createTextBlock = (id, text, x, y, fontSize) => ({
      id,
      type: 'text', // Явно указываем тип
      text,
      x,
      y,
      fontSize,
      color: getInverseColor(x, y, 100, fontSize + 10) // берём область под текст ~100px шириной
    });

    textBlocks.push(createTextBlock("name", employee.full_name, 20, 20, 24));
    textBlocks.push(createTextBlock("position", employee.position, 20, 60, 20));

    if (privacyLevel === "medium" || privacyLevel === "high") {
      textBlocks.push(createTextBlock("company", employee.company, 20, 100, 18));
      textBlocks.push(createTextBlock("department", employee.department, 20, 140, 18));
      textBlocks.push(createTextBlock("location", employee.office_location, 20, 180, 18));
    }

    if (privacyLevel === "high") {
      textBlocks.push(createTextBlock("email", `Email: ${employee.contact.email}`, 20, 220, 16));
      textBlocks.push(createTextBlock("telegram", `Telegram: ${employee.contact.telegram}`, 20, 260, 16));
    }

    // Добавляем логотип, если он есть в branding
    const allBlocks = [bgBlock, ...textBlocks];
    if (employee.branding && employee.branding.logo_url) {
      const logoBlock = {
        id: "logo",
        type: "image",
        imageSrc: employee.branding.logo_url,
        x: 1020, // Справа (1280 - 240 = 1040, минус 20 отступ)
        y: 20,   // Сверху
        width: 240,
        height: 150,
        objectFit: "contain"
      };
      allBlocks.push(logoBlock);
    }

    setBlocks(allBlocks);
  }, [bgBlockData, blocks, setBlocks]);

  const handleUpdate = (id, newProps) => {
    setBlocks((prev) => prev.map((b) => (b.id === id ? { ...b, ...newProps } : b)));
  };

  const handleDelete = (id) => {
    setBlocks((prev) => {
      const filtered = prev.filter((b) => b.id !== id);
      // Если удаляется блок "b1", сбрасываем lastEmployeeRef
      if (id === "b1") {
        lastEmployeeRef.current = null;
      }
      return filtered;
    });
    setSelectedBlockId(null);
  };

  const handleEditorClose = (deleteId, shouldDelete) => {
    if (shouldDelete && deleteId) {
      handleDelete(deleteId);
    }
    setSelectedBlockId(null);
  };

  const handleTextSelect = (id) => {
    textClickedRef.current = true;
    setSelectedBlockId(id);
    // Сбрасываем флаг через увеличенную задержку для надёжности
    setTimeout(() => {
      textClickedRef.current = false;
    }, 50);
  };

  const handleBackgroundClick = (e) => {
    // Не закрываем панель, если только что кликнули по тексту
    if (textClickedRef.current) {
      return;
    }
    
    // Закрываем панель только при клике на фоновый слой или контейнер
    if (e.target === containerRef.current || e.target === backgroundLayerRef.current) {
      setSelectedBlockId(null);
    }
  };

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
      
      {/* Слой 1: Фоновое изображение или реальное видео */}
      <div
        ref={backgroundLayerRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          // Если есть загруженный фон - используем его, иначе реальное видео (через canvas)
          backgroundImage: backgroundImage ? `url(${backgroundImage})` : 'none',
          backgroundSize: "cover",
          backgroundPosition: "center",
          // Размытие применяется только к загруженному фону через CSS
          filter: (backgroundImage && backgroundBlur > 0) ? `blur(${backgroundBlur}px)` : 'none',
          zIndex: 0
        }}
      />

      {/* Слой 2: Текстовые блоки и изображения */}
      {blocks.map((b) => {
        // Пропускаем блок метаданных (b1)
        if (b.type === 'metadata' || b.id === 'b1') {
          return null;
        }
        
        // Рендерим изображение или текст в зависимости от типа блока
        if (b.type === 'image') {
          return (
            <DraggableImage
              key={b.id}
              block={b}
              selected={b.id === selectedBlockId}
              onSelect={handleTextSelect}
              onUpdate={handleUpdate}
              parentRef={containerRef}
            />
          );
        }
        
        // По умолчанию рендерим как текст
        return (
          <DraggableText
            key={b.id}
            block={b}
            selected={b.id === selectedBlockId}
            onSelect={handleTextSelect}
            onUpdate={handleUpdate}
            parentRef={containerRef}
            backgroundLayerRef={backgroundLayerRef}
          />
        );
      })}

      {/* Слой 3: Canvas с веб-камерой (прозрачный фон в обычном режиме) */}
      <canvas
        ref={canvasRef}
        style={{ 
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%", 
          height: "100%", 
          zIndex: 2,
          pointerEvents: rawMode ? "auto" : "none" // В rawMode разрешаем клики, в обработанном - пропускаем к тексту
        }}
      />

      {/* Overlay и панель редактирования */}
      {selectedBlockId && blocks.find(b => b.id === selectedBlockId) && (() => {
        const selectedBlock = blocks.find(b => b.id === selectedBlockId);
        const EditorPanel = selectedBlock.type === 'image' ? ImageEditorPanel : TextEditorPanel;
        
        return (
          <>
            <div 
              className="text-editor-overlay" 
              onClick={() => setSelectedBlockId(null)}
            />
            <EditorPanel
              key={selectedBlockId}
              block={selectedBlock}
              onUpdate={handleUpdate}
              onClose={handleEditorClose}
            />
          </>
        );
      })()}
    </div>
  );
}