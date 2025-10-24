import React, { useRef, useEffect, useState } from "react";
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import DraggableText from "./DraggableText";

// Global model management
let globalModel = null;
let modelCreationInProgress = false;

export default function WebcamWithText({ blocks, setBlocks, selectedBlockId, setSelectedBlockId, onStatsUpdate }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [model, setModel] = useState(null);
  const recRef = useRef([]);
  const backgroundRef = useRef(null);
  const frameCountRef = useRef(0);
  const totalTimeRef = useRef(0);
  const lastStatsUpdateRef = useRef(0);
  
  const [webGPUSupported, setWebGPUSupported] = useState(false);
  const [backendInfo, setBackendInfo] = useState('Checking...');

  const MODEL_SCALE = 1; // 0.5 = вход 640x480, 1.0 = вход 1280x960
  const downsampleRatioQuality = 1; // Внутренний downsampling модели для ускорения88

  useEffect(() => {
    const initializeModel = async () => {
      if (modelCreationInProgress) {
        console.log("🔄 Model loading already in progress, waiting...");
        return;
      }

      modelCreationInProgress = true;

      try {
        // Get camera stream first
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        // Use existing global model if available
        if (globalModel) {
          console.log("♻️ Reusing existing global model");
          setModel(globalModel);
          modelCreationInProgress = false;
          return;
        }

        // Set backend to WebGPU
        console.log("🔧 Setting TensorFlow.js backend to WebGPU...");
        try {
          await tf.setBackend('webgpu');
          await tf.ready();
          console.log("✅ WebGPU backend initialized");
          setWebGPUSupported(true);
          setBackendInfo('WebGPU (TensorFlow.js)');
        } catch (error) {
          console.warn("WebGPU not available, falling back to WebGL:", error);
          await tf.setBackend('webgl');
          await tf.ready();
          console.log("✅ WebGL backend initialized");
          setWebGPUSupported(false);
          setBackendInfo('WebGL (TensorFlow.js)');
        }

        console.log("🔄 Loading TensorFlow.js model...");
        const loadedModel = await tf.loadGraphModel('/rvm_mobilenetv3_tfjs_int8/model.json');
        console.log("✅ Model loaded successfully!");
        console.log("🎮 Backend:", tf.getBackend());
        
        // Debug: print model inputs and outputs (закомментировано)
        // console.log("📋 Model inputs:", loadedModel.inputs.map(i => `${i.name} ${i.shape}`));
        // console.log("📋 Model outputs:", loadedModel.outputs.map(o => `${o.name} ${o.shape}`));
        
        globalModel = loadedModel;
        setModel(loadedModel);
        
        // Warmup: run model once to initialize recurrent states with correct sizes
        // Используем MODEL_SCALE для определения размеров
        const origWidth = 1280;
        const origHeight = 960;
        const modelInputWidth = Math.round(origWidth * MODEL_SCALE);
        const modelInputHeight = Math.round(origHeight * MODEL_SCALE);
        
        console.log("🔥 Running warmup to initialize recurrent states...");
        console.log(`📐 Model input size: ${modelInputWidth}x${modelInputHeight} (scale: ${MODEL_SCALE})`);
        console.log(`⚙️ downsampleRatio: ${downsampleRatioQuality}`);
        
        const dummyInput = tf.zeros([1, modelInputHeight, modelInputWidth, 3]); // Dummy video frame
        const dummyRec = [
          tf.zeros([1, 1, 1, 1]),
          tf.zeros([1, 1, 1, 1]),
          tf.zeros([1, 1, 1, 1]),
          tf.zeros([1, 1, 1, 1])
        ];
        const dummyRatio = tf.tensor1d([downsampleRatioQuality]);
        
        try {
          const warmupOutputs = await loadedModel.executeAsync({
            'src': dummyInput,
            'r1i': dummyRec[0],
            'r2i': dummyRec[1],
            'r3i': dummyRec[2],
            'r4i': dummyRec[3],
            'downsample_ratio': dummyRatio
          });
          
          // ОТЛАДКА: выводим ВСЕ выходы (закомментировано для чистоты консоли)
          // console.log("🔍 Warmup outputs count:", warmupOutputs.length);
          // warmupOutputs.forEach((output, i) => {
          //   console.log(`  Output[${i}] shape:`, output.shape);
          // });
          
          // Анализируем выходы по размерам:
          // fgr должна быть [1, H, W, 3]
          // pha должна быть [1, H, W, 1]
          // r1-r4 - меньшие размеры
          
          let fgrIdx = -1, phaIdx = -1;
          const recIndices = [];
          
          warmupOutputs.forEach((output, i) => {
            const shape = output.shape;
            if (shape.length === 4) {
              if (shape[3] === 3) {
                fgrIdx = i; // foreground RGB
              } else if (shape[3] === 1) {
                phaIdx = i; // alpha mask
              } else {
                recIndices.push(i); // recurrent states
              }
            }
          });
          
          // console.log(`📍 Detected: fgr at [${fgrIdx}], pha at [${phaIdx}], rec at [${recIndices}]`);
          
          // Сортируем рекуррентные состояния по размеру (от большего к меньшему)
          // r4 > r3 > r2 > r1 (обычно)
          recIndices.sort((a, b) => {
            const sizeA = warmupOutputs[a].shape.reduce((acc, val) => acc * val, 1);
            const sizeB = warmupOutputs[b].shape.reduce((acc, val) => acc * val, 1);
            return sizeB - sizeA; // от большего к меньшему
          });
          
          // Присваиваем рекуррентные состояния в порядке r1, r2, r3, r4
          // recIndices отсортированы от большего к меньшему
          // ВАЖНО: r1i должен быть самым БОЛЬШИМ, r4i - самым МАЛЕНЬКИМ (проверено по ошибке!)
          recRef.current = [
            warmupOutputs[recIndices[0]], // r1 - САМЫЙ БОЛЬШОЙ
            warmupOutputs[recIndices[1]], // r2
            warmupOutputs[recIndices[2]], // r3
            warmupOutputs[recIndices[3]]  // r4 - САМЫЙ МАЛЕНЬКИЙ
          ];
          
          // Dispose fgr and pha
          if (fgrIdx >= 0) warmupOutputs[fgrIdx].dispose();
          if (phaIdx >= 0) warmupOutputs[phaIdx].dispose();
          
          console.log("✅ Warmup complete, recurrent states initialized");
          
          // Проверяем количество тензоров в памяти
          console.log(`📊 Tensors in memory: ${tf.memory().numTensors}`);
          console.log(`💾 Memory usage: ${(tf.memory().numBytes / 1024 / 1024).toFixed(2)} MB`);
        } catch (warmupError) {
          console.error("❌ Warmup failed:", warmupError);
          // Fallback to zeros
          recRef.current = [
            tf.zeros([1, 1, 1, 1]),
            tf.zeros([1, 1, 1, 1]),
            tf.zeros([1, 1, 1, 1]),
            tf.zeros([1, 1, 1, 1])
          ];
        }
        
        // Dispose warmup tensors
        dummyInput.dispose();
        dummyRec.forEach(t => t.dispose());
        dummyRatio.dispose();

        // Create background
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

      } catch (error) {
        console.error("❌ Model initialization failed:", error);
        setBackendInfo('Failed to load');
      } finally {
        modelCreationInProgress = false;
      }
    };

    initializeModel();

    return () => {
      // Cleanup
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!model) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let isProcessing = false;

    const processFrame = async () => {
      if (isProcessing) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      isProcessing = true;
      const startTime = performance.now();

      try {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
          const origWidth = 1280;
          const origHeight = 960;
          const modelInputWidth = Math.round(origWidth * MODEL_SCALE);
          const modelInputHeight = Math.round(origHeight * MODEL_SCALE);
          
          // Resize video to model input size and convert to tensor
          const inputTensor = tf.tidy(() => {
            // Capture video frame as tensor
            const videoTensor = tf.browser.fromPixels(video);
            // Resize to model input size
            const resized = tf.image.resizeBilinear(videoTensor, [modelInputHeight, modelInputWidth]);
            // Normalize to [0, 1]
            const normalized = resized.div(255.0);
            // Add batch dimension [1, H, W, 3]
            const batched = normalized.expandDims(0);
            return batched;
          });

          const modelStartTime = performance.now();

          // Run model inference
          // Model inputs: src, r1i, r2i, r3i, r4i, downsample_ratio
          // Model outputs: pha, fgr, r1o, r2o, r3o, r4o
          const downsampleRatioTensor = tf.tensor1d([downsampleRatioQuality]);
          
          const outputs = await model.executeAsync({
            'src': inputTensor,
            'r1i': recRef.current[0],
            'r2i': recRef.current[1],
            'r3i': recRef.current[2],
            'r4i': recRef.current[3],
            'downsample_ratio': downsampleRatioTensor
          });

          const modelInferenceTime = performance.now() - modelStartTime;

          // Extract outputs - используем ту же логику что и в warmup
          let phaTensor, fgrTensor;
          const recOutputs = [];
          
          outputs.forEach((output, i) => {
            const shape = output.shape;
            if (shape.length === 4) {
              if (shape[3] === 3) {
                fgrTensor = output; // foreground RGB
              } else if (shape[3] === 1) {
                phaTensor = output; // alpha mask
              } else {
                recOutputs.push(output); // recurrent states
              }
            }
          });
          
          // Сортируем рекуррентные выходы по размеру (от большего к меньшему)
          recOutputs.sort((a, b) => {
            const sizeA = a.shape.reduce((acc, val) => acc * val, 1);
            const sizeB = b.shape.reduce((acc, val) => acc * val, 1);
            return sizeB - sizeA;
          });

          // Сохраняем старые состояния для удаления
          const oldStates = recRef.current;
          
          // Update recurrent states
          // ВАЖНО: r1 = самый БОЛЬШОЙ, r4 = самый МАЛЕНЬКИЙ
          recRef.current = [
            recOutputs[0], // r1 - САМЫЙ БОЛЬШОЙ
            recOutputs[1], // r2
            recOutputs[2], // r3
            recOutputs[3]  // r4 - САМЫЙ МАЛЕНЬКИЙ
          ];
          
          // НЕ удаляем oldStates вручную - TensorFlow.js управляет памятью сам
          // oldStates.forEach(t => t.dispose());
          
          // Dispose fgr
          if (fgrTensor) fgrTensor.dispose();

          // Get mask data
          const phaData = await phaTensor.data();
          const phaShape = phaTensor.shape; // [1, height, width, 1] or [1, 1, height, width]
          
          let actualMaskHeight, actualMaskWidth;
          if (phaShape.length === 4) {
            if (phaShape[3] === 1) {
              // NHWC format: [1, height, width, 1]
              actualMaskHeight = phaShape[1];
              actualMaskWidth = phaShape[2];
            } else {
              // NCHW format: [1, 1, height, width]
              actualMaskHeight = phaShape[2];
              actualMaskWidth = phaShape[3];
            }
          }
          
          // Dispose phaTensor сразу после получения данных
          phaTensor.dispose();

          // Draw mask to canvas
          const maskCanvas = document.createElement('canvas');
          maskCanvas.width = actualMaskWidth;
          maskCanvas.height = actualMaskHeight;
          const maskCtx = maskCanvas.getContext('2d');
          const maskImageData = maskCtx.createImageData(actualMaskWidth, actualMaskHeight);

          // Fill mask
          for (let i = 0; i < actualMaskWidth * actualMaskHeight; i++) {
            const alpha = Math.min(1, Math.max(0, phaData[i]));
            const alphaVal = alpha * 255;
            maskImageData.data[i * 4] = alphaVal;
            maskImageData.data[i * 4 + 1] = alphaVal;
            maskImageData.data[i * 4 + 2] = alphaVal;
            maskImageData.data[i * 4 + 3] = 255;
          }

          maskCtx.putImageData(maskImageData, 0, 0);

          // Scale mask to original size
          const fullMaskCanvas = document.createElement('canvas');
          fullMaskCanvas.width = origWidth;
          fullMaskCanvas.height = origHeight;
          const fullMaskCtx = fullMaskCanvas.getContext('2d');
          fullMaskCtx.imageSmoothingEnabled = true;
          fullMaskCtx.imageSmoothingQuality = 'high';
          fullMaskCtx.drawImage(maskCanvas, 0, 0, origWidth, origHeight);
          const fullMaskData = fullMaskCtx.getImageData(0, 0, origWidth, origHeight);

          // Get original video frame
          ctx.drawImage(video, 0, 0, origWidth, origHeight);
          const originalImageData = ctx.getImageData(0, 0, origWidth, origHeight);

          // Composite with background
          const background = backgroundRef.current;
          const outputData = new Uint8ClampedArray(origWidth * origHeight * 4);

          for (let i = 0; i < origWidth * origHeight; i++) {
            const i4 = i * 4;
            const alpha = fullMaskData.data[i4] / 255.0;
            const oneMinusAlpha = 1 - alpha;

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

          // Draw final image
          const outputImageData = new ImageData(outputData, origWidth, origHeight);
          ctx.putImageData(outputImageData, 0, 0);

          // Dispose tensors (НЕ удаляем phaTensor и fgrTensor - они уже удалены выше)
          inputTensor.dispose();
          downsampleRatioTensor.dispose();
          // phaTensor и fgrTensor уже удалены после обработки
          // recurrent states сохранены в recRef.current, старые удалены

          // Calculate FPS and update stats
          const frameTime = performance.now() - startTime;
          totalTimeRef.current += frameTime;
          frameCountRef.current += 1;
          const fps = 1000.0 / frameTime;
          const avgFps = (frameCountRef.current * 1000.0) / totalTimeRef.current;

          const now = performance.now();
          if (onStatsUpdate && (now - lastStatsUpdateRef.current) > 100) {
            lastStatsUpdateRef.current = now;
            onStatsUpdate({
              fps: fps.toFixed(2),
              avgFps: avgFps.toFixed(2),
              modelTime: modelInferenceTime.toFixed(2),
              fullFrameTime: frameTime.toFixed(2),
              modelActive: true,
              backend: backendInfo
            });
          }
        }
      } catch (error) {
        console.error("❌ Frame processing error:", error);
        // Fallback: just draw the video
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }

      isProcessing = false;
      animationFrameId = requestAnimationFrame(processFrame);
    };

    processFrame();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [model, onStatsUpdate, backendInfo]);

  return (
    <div ref={containerRef} style={{ position: "relative", display: "inline-block" }}>
      <video ref={videoRef} style={{ display: "none" }} width="1280" height="960" />
      <canvas ref={canvasRef} width="1280" height="960" style={{ display: "block", borderRadius: "8px" }} />
      
      {blocks.map((block) => (
        <DraggableText
          key={block.id}
          block={block}
          blocks={blocks}
          setBlocks={setBlocks}
          selectedBlockId={selectedBlockId}
          setSelectedBlockId={setSelectedBlockId}
          containerRef={containerRef}
        />
      ))}
    </div>
  );
}

