import React, { useState } from "react";

export default function DraggableText({ block, onUpdate, selected, onSelect, parentRef }) {
  const [dragging, setDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e) => {
    e.stopPropagation();
    setDragging(true);
    onSelect(block.id);
    const parentRect = parentRef.current.getBoundingClientRect();
    setOffset({
      x: e.clientX - parentRect.left - block.x,
      y: e.clientY - parentRect.top - block.y
    });
  };

  const handleMouseMove = (e) => {
    if (!dragging) return;
    const parentRect = parentRef.current.getBoundingClientRect();
    const newX = e.clientX - parentRect.left - offset.x;
    const newY = e.clientY - parentRect.top - offset.y;
    onUpdate(block.id, { x: newX, y: newY });
  };

  const handleMouseUp = () => {
    setDragging(false);
  };

  return (
    <div
      style={{
        position: "absolute",
        left: block.x,
        top: block.y,
        fontSize: block.fontSize,
        color: block.color,
        cursor: "move",
        userSelect: "none",
        border: selected ? "1px dashed white" : "none",
        padding: "2px"
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {block.text}
    </div>
  );
}