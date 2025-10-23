import React, { useRef, useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import DraggableText from "./DraggableText";

// Конвертация float32 в float16
function floatToFloat16(value) {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const x = int32View[0];
  let bits = (x >> 16) & 0x8000;
  let m = (x >> 12) & 0x07ff;
  const e = (x >> 23) & 0xff;
  if (e < 103) return bits;
  if (e > 142) {
    bits |= 0x7c00;
    bits |= ((e === 255) ? 0 : 1) && (x & 0x007fffff);
    return bits;
  }
  if (e < 113) {
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

export default function WebcamWithText({ blocks, setBlocks, selectedBlockId, setSelectedBlockId }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [session, setSession] = useState(null);
  const recRef = useRef([]);
  const backgroundRef = useRef(null);
  const frameCountRef = useRef(0);
  const totalTimeRef = useRef(0);

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

    // Загрузка ONNX модели
    ort.InferenceSession.create("/rvm_mobilenetv3_fp16.onnx", {
      executionProviders: ['webgl', 'wasm']
    }).then((sess) => {
      console.log("Model loaded, execution providers:", sess.inputNames);
      setSession(sess);
      // Инициализация recurrent states - должны быть float16
      recRef.current = [
        new ort.Tensor("float16", new Uint16Array(1), [1, 1, 1, 1]),
        new ort.Tensor("float16", new Uint16Array(1), [1, 1, 1, 1]),
        new ort.Tensor("float16", new Uint16Array(1), [1, 1, 1, 1]),
        new ort.Tensor("float16", new Uint16Array(1), [1, 1, 1, 1])
      ];
    }).catch(console.error);

    // Создание фона
    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = 640;
    bgCanvas.height = 480;
    const bgCtx = bgCanvas.getContext('2d');
    const gradient = bgCtx.createLinearGradient(0, 0, 640, 480);
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#16213e');
    bgCtx.fillStyle = gradient;
    bgCtx.fillRect(0, 0, 640, 480);
    backgroundRef.current = bgCtx.getImageData(0, 0, 640, 480);
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

          // Конвертация в float16 для src
          const float16Data = new Uint16Array(rgbData.length);
          for (let i = 0; i < rgbData.length; i++) {
            float16Data[i] = floatToFloat16(rgbData[i]);
          }

          const inputTensor = new ort.Tensor("float16", float16Data, [1, 3, height, width]);
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

            const results = await session.run(feeds);
            
            // Извлечение результатов
            const fgr = results.fgr.data;
            const pha = results.pha.data;
            
            // Обновление rec states
            if (results.r1o) recRef.current[0] = results.r1o;
            if (results.r2o) recRef.current[1] = results.r2o;
            if (results.r3o) recRef.current[2] = results.r3o;
            if (results.r4o) recRef.current[3] = results.r4o;

            // Композитинг с фоном
            const background = backgroundRef.current;
            const outputData = new Uint8ClampedArray(width * height * 4);
            
            for (let i = 0; i < width * height; i++) {
              const alpha = pha[i];
              const r = Math.min(255, Math.max(0, fgr[i] * 255));
              const g = Math.min(255, Math.max(0, fgr[width * height + i] * 255));
              const b = Math.min(255, Math.max(0, fgr[2 * width * height + i] * 255));
              
              const bgR = background ? background.data[i * 4] : 26;
              const bgG = background ? background.data[i * 4 + 1] : 26;
              const bgB = background ? background.data[i * 4 + 2] : 46;
              
              outputData[i * 4] = r * alpha + bgR * (1 - alpha);
              outputData[i * 4 + 1] = g * alpha + bgG * (1 - alpha);
              outputData[i * 4 + 2] = b * alpha + bgB * (1 - alpha);
              outputData[i * 4 + 3] = 255;
            }

            const outputImageData = new ImageData(outputData, width, height);
            ctx.putImageData(outputImageData, 0, 0);

            // Расчет FPS
            const frameTime = performance.now() - startTime;
            totalTimeRef.current += frameTime;
            frameCountRef.current += 1;
            const fps = 1000.0 / frameTime;
            const avgFps = (frameCountRef.current * 1000.0) / totalTimeRef.current;

            ctx.fillStyle = 'red';
            ctx.font = '20px Arial';
            ctx.fillText(`FPS: ${fps.toFixed(2)}`, 10, 30);
            ctx.fillStyle = 'orange';
            ctx.font = '16px Arial';
            ctx.fillText(`Avg FPS: ${avgFps.toFixed(2)}`, 10, 55);
            ctx.fillStyle = 'yellow';
            ctx.fillText(`Frame time: ${frameTime.toFixed(2)} ms`, 10, 80);
          } catch (error) {
            console.error("Model inference error:", error);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          }
        } else {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
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
        width: "640px",
        height: "480px",
        border: "1px solid #333",
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
