import React, { useRef, useEffect } from "react";
import DraggableText from "./DraggableText";

export default function WebcamWithText({ blocks, setBlocks, selectedBlockId, setSelectedBlockId }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

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
  }, []);

  useEffect(() => {
    const drawFrame = () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      // Проверяем, что видео уже имеет размеры
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }

      requestAnimationFrame(drawFrame);
    };

    requestAnimationFrame(drawFrame);

    return () => cancelAnimationFrame(drawFrame);
  }, []);

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
