import React, { useState, useEffect, useRef } from "react";

export default function DraggableText({ block, onUpdate, selected, onSelect, parentRef }) {
  const [dragging, setDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragStateRef = useRef({ 
    isDragging: false, 
    offsetX: 0, 
    offsetY: 0,
    hasMoved: false,
    startX: 0,
    startY: 0
  });

  const handleMouseDown = (e) => {
    e.stopPropagation();
    e.preventDefault();
    
    const parentRect = parentRef.current.getBoundingClientRect();
    const offsetX = e.clientX - parentRect.left - block.x;
    const offsetY = e.clientY - parentRect.top - block.y;
    
    setOffset({ x: offsetX, y: offsetY });
    setDragging(true);
    
    // Сохраняем начальную позицию для определения, был ли это клик или drag
    dragStateRef.current = { 
      isDragging: true, 
      offsetX, 
      offsetY,
      hasMoved: false,
      startX: e.clientX,
      startY: e.clientY
    };
  };

  // Глобальные обработчики для предотвращения срыва при быстром движении
  useEffect(() => {
    const handleGlobalMouseMove = (e) => {
      if (!dragStateRef.current.isDragging || !parentRef.current) return;
      
      // Проверяем, сдвинулась ли мышь больше чем на 5 пикселей
      const deltaX = Math.abs(e.clientX - dragStateRef.current.startX);
      const deltaY = Math.abs(e.clientY - dragStateRef.current.startY);
      
      if (deltaX > 5 || deltaY > 5) {
        dragStateRef.current.hasMoved = true;
      }
      
      // Обновляем позицию только если было движение
      if (dragStateRef.current.hasMoved) {
        const parentRect = parentRef.current.getBoundingClientRect();
        const newX = e.clientX - parentRect.left - dragStateRef.current.offsetX;
        const newY = e.clientY - parentRect.top - dragStateRef.current.offsetY;
        
        onUpdate(block.id, { x: newX, y: newY });
      }
    };

    const handleGlobalMouseUp = () => {
      if (dragStateRef.current.isDragging) {
        // Если не было движения - это клик, открываем редактор
        if (!dragStateRef.current.hasMoved) {
          onSelect(block.id);
        }
        
        setDragging(false);
        dragStateRef.current.isDragging = false;
        dragStateRef.current.hasMoved = false;
      }
    };

    if (dragging) {
      document.addEventListener('mousemove', handleGlobalMouseMove);
      document.addEventListener('mouseup', handleGlobalMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleGlobalMouseMove);
      document.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [dragging, block.id, onUpdate, onSelect, parentRef]);

  // Курсор grabbing только если реально перетаскиваем
  const cursorStyle = (dragging && dragStateRef.current.hasMoved) ? "grabbing" : "grab";

  return (
    <div
      style={{
        position: "absolute",
        left: block.x,
        top: block.y,
        fontSize: `${block.fontSize}px`,
        color: block.color,
        fontFamily: block.fontFamily || 'Arial',
        fontWeight: block.fontWeight || 'normal',
        fontStyle: block.fontStyle || 'normal',
        cursor: cursorStyle,
        userSelect: "none",
        border: selected ? "2px solid #2a5298" : "2px solid transparent",
        padding: "4px 8px",
        borderRadius: "4px",
        backgroundColor: selected ? "rgba(42, 82, 152, 0.1)" : "transparent",
        transition: selected ? "none" : "all 0.2s ease",
        whiteSpace: "nowrap",
        textShadow: "1px 1px 2px rgba(0,0,0,0.5)",
        zIndex: 1 // Текст между фоном (0) и canvas (2)
      }}
      onMouseDown={handleMouseDown}
    >
      {block.text}
    </div>
  );
}