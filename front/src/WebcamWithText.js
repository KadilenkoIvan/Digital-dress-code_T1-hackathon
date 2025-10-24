import React, { useRef, useEffect, useState } from "react";
import * as ort from 'onnxruntime-web/webgpu';
import DraggableText from "./DraggableText";

// Global session management to prevent multiple sessions
let globalSession = null;
let sessionCreationInProgress = false;

export default function WebcamWithText({ blocks, setBlocks, selectedBlockId, setSelectedBlockId, onStatsUpdate }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [session, setSession] = useState(null);
  const recRef = useRef([]);
  const backgroundRef = useRef(null);
  const frameCountRef = useRef(0);
  const totalTimeRef = useRef(0);
  const lastStatsUpdateRef = useRef(0);
  const downsampleCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const fullMaskCanvasRef = useRef(null);
  const originalVideoCanvasRef = useRef(null); // Новый canvas для оригинального видео
  
  const [webGPUSupported, setWebGPUSupported] = useState(false);
  const [backendInfo, setBackendInfo] = useState('Checking...');
  const [usingWebGPU, setUsingWebGPU] = useState(false);

  const MODEL_SCALE = 0.4;
  const downsampleRatioQuality = 0.6;

  useEffect(() => {
    const checkWebGPUSupport = async () => {
      try {
        if (navigator.gpu) {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            setWebGPUSupported(true);
            setBackendInfo('WebGPU');
            setUsingWebGPU(true);
            console.log("✅ WebGPU is supported");
            return true;
          }
        }
      } catch (error) {
        console.warn("WebGPU not supported:", error);
      }
      
      setWebGPUSupported(false);
      setBackendInfo('WASM (CPU)');
      setUsingWebGPU(false);
      console.log("⚠️ WebGPU not available, using WASM");
      return false;
    };

    const initializeSession = async () => {
      if (sessionCreationInProgress) {
        console.log("🔄 Session creation already in progress, waiting...");
        return;
      }

      sessionCreationInProgress = true;

      try {
        const isWebGPUAvaliable = await checkWebGPUSupport();
        
        // Create temporary canvases
        downsampleCanvasRef.current = document.createElement('canvas');
        maskCanvasRef.current = document.createElement('canvas');
        fullMaskCanvasRef.current = document.createElement('canvas');
        originalVideoCanvasRef.current = document.createElement('canvas'); // Canvas для оригинального видео
        
        // Get camera stream
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((stream) => {
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
              videoRef.current.play().catch(console.error);
            }
          })
          .catch(console.error);

        // Use existing global session if available
        if (globalSession) {
          console.log("♻️ Reusing existing global session");
          setSession(globalSession);
          sessionCreationInProgress = false;
          return;
        }

        // Configure ONNX Runtime
        if (isWebGPUAvaliable) {
          try {
            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter.requestDevice();
            ort.env.webgpu.device = device;
          } catch (error) {
            console.warn("Failed to initialize WebGPU device:", error);
            setWebGPUSupported(false);
            setBackendInfo('WASM (Fallback)');
            setUsingWebGPU(false);
          }
        } else {
          ort.env.wasm.numThreads = 1;
          ort.env.wasm.simd = true;
        }

        // Load ONNX model with proper error handling
        console.log("🔄 Loading model...");
        const executionProviders = isWebGPUAvaliable && ort.env.webgpu.device ? 
          ['webgpu', 'wasm'] : ['wasm'];

        try {
          const sess = await ort.InferenceSession.create("/rvm_mobilenetv3_fp32.onnx", {
            executionProviders: executionProviders,
            graphOptimizationLevel: 'all'
          });
          
          console.log("✅ Model loaded successfully!");
          console.log("🎮 Backend:", executionProviders[0]);
          
          globalSession = sess;
          setSession(sess);
          
          // Initialize recurrent states
          recRef.current = [
            new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
            new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
            new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1]),
            new ort.Tensor("float32", new Float32Array(1).fill(0), [1, 1, 1, 1])
          ];

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

        } catch (modelError) {
          console.error("❌ Model loading failed:", modelError);
          
          if (isWebGPUAvaliable) {
            console.log("🔄 Falling back to WASM...");
            const sess = await ort.InferenceSession.create("/rvm_mobilenetv3_fp32.onnx", {
              executionProviders: ['wasm']
            });
            globalSession = sess;
            setSession(sess);
            setBackendInfo('WASM (Fallback)');
            setUsingWebGPU(false);
          } else {
            throw modelError;
          }
        }
      } catch (error) {
        console.error("❌ Session initialization failed:", error);
        setBackendInfo('Failed to load');
      } finally {
        sessionCreationInProgress = false;
      }
    };

    initializeSession();

    return () => {
      // Don't cleanup global session on component unmount
    };
  }, []);

  // Separate useEffect for the rendering loop
  useEffect(() => {
    let animationId;
    let isProcessing = false;

    const drawFrame = async () => {
      if (!videoRef.current || !canvasRef.current || isProcessing) {
        animationId = requestAnimationFrame(drawFrame);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        // Set canvas dimensions only once
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        if (session && recRef.current.length > 0) {
          isProcessing = true;
          const startTime = performance.now();

          try {
            // Original dimensions
            const origWidth = canvas.width;
            const origHeight = canvas.height;
            
            // Model dimensions
            const modelWidth = Math.round(origWidth * MODEL_SCALE);
            const modelHeight = Math.round(origHeight * MODEL_SCALE);
            
            // Prepare downsampled image for model
            const downsampleCanvas = downsampleCanvasRef.current;
            downsampleCanvas.width = modelWidth;
            downsampleCanvas.height = modelHeight;
            const downsampleCtx = downsampleCanvas.getContext('2d');
            
            downsampleCtx.drawImage(video, 0, 0, modelWidth, modelHeight);
            const imageData = downsampleCtx.getImageData(0, 0, modelWidth, modelHeight);
            
            const rgbData = new Float32Array(3 * modelWidth * modelHeight);
            
            // Convert to RGB and normalize [0, 1]
            for (let i = 0; i < modelWidth * modelHeight; i++) {
              rgbData[i] = imageData.data[i * 4] / 255.0;
              rgbData[modelWidth * modelHeight + i] = imageData.data[i * 4 + 1] / 255.0;
              rgbData[2 * modelWidth * modelHeight + i] = imageData.data[i * 4 + 2] / 255.0;
            }

            // Create input tensor
            const inputTensor = new ort.Tensor("float32", rgbData, [1, 3, modelHeight, modelWidth]);
            const downsampleRatio = new ort.Tensor("float32", new Float32Array([downsampleRatioQuality]), [1]);

            // Prepare feeds
            const feeds = {
              src: inputTensor,
              r1i: recRef.current[0],
              r2i: recRef.current[1],
              r3i: recRef.current[2],
              r4i: recRef.current[3],
              downsample_ratio: downsampleRatio
            };

            // Run model
            const modelStartTime = performance.now();
            const results = await session.run(feeds);
            const modelInferenceTime = performance.now() - modelStartTime;
            
            // Extract results
            const phaSmall = results.pha.data;
            
            // Update recurrent states
            if (results.r1o) recRef.current[0] = results.r1o;
            if (results.r2o) recRef.current[1] = results.r2o;
            if (results.r3o) recRef.current[2] = results.r3o;
            if (results.r4o) recRef.current[3] = results.r4o;

            // Get original video frame without drawing to main canvas
            const originalVideoCanvas = originalVideoCanvasRef.current;
            originalVideoCanvas.width = origWidth;
            originalVideoCanvas.height = origHeight;
            const originalVideoCtx = originalVideoCanvas.getContext('2d');
            originalVideoCtx.drawImage(video, 0, 0, origWidth, origHeight);
            const originalImageData = originalVideoCtx.getImageData(0, 0, origWidth, origHeight);
            
            // Process mask
            const maskCanvas = maskCanvasRef.current;
            maskCanvas.width = modelWidth;
            maskCanvas.height = modelHeight;
            const maskCtx = maskCanvas.getContext('2d');
            const maskImageData = maskCtx.createImageData(modelWidth, modelHeight);
            
            // Fill mask (grayscale)
            for (let i = 0; i < modelWidth * modelHeight; i++) {
              const alpha = Math.min(1, Math.max(0, phaSmall[i]));
              const alphaVal = alpha * 255;
              maskImageData.data[i * 4] = alphaVal;
              maskImageData.data[i * 4 + 1] = alphaVal;
              maskImageData.data[i * 4 + 2] = alphaVal;
              maskImageData.data[i * 4 + 3] = 255;
            }
            
            maskCtx.putImageData(maskImageData, 0, 0);
            
            // Scale mask to original size
            const fullMaskCanvas = fullMaskCanvasRef.current;
            fullMaskCanvas.width = origWidth;
            fullMaskCanvas.height = origHeight;
            const fullMaskCtx = fullMaskCanvas.getContext('2d');
            fullMaskCtx.drawImage(maskCanvas, 0, 0, origWidth, origHeight);
            const fullMaskData = fullMaskCtx.getImageData(0, 0, origWidth, origHeight);

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

            // Draw the final composited image to main canvas
            const outputImageData = new ImageData(outputData, origWidth, origHeight);
            ctx.putImageData(outputImageData, 0, 0);

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

            console.log(`🔄 Frame processed: ${fps.toFixed(1)} FPS, model: ${modelInferenceTime.toFixed(1)}ms`);
          } catch (error) {
            console.error("❌ Model inference error:", error);
            // Fallback: just draw the video
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
                backend: backendInfo
              });
            }
          } finally {
            isProcessing = false;
          }
        } else {
          // No session available, just draw video
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
              backend: backendInfo
            });
          }
        }
      } else {
        // Video not ready yet, clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      animationId = requestAnimationFrame(drawFrame);
    };

    animationId = requestAnimationFrame(drawFrame);

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [session, backendInfo, onStatsUpdate]);

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
      <div style={{
        position: "absolute",
        top: "10px",
        left: "10px",
        background: "rgba(0,0,0,0.7)",
        color: "white",
        padding: "5px 10px",
        borderRadius: "5px",
        fontSize: "14px",
        zIndex: 1000
      }}>
        Backend: {backendInfo}
      </div>
      
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