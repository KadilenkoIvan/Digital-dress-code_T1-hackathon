import React, { useRef, useState, useEffect } from "react";
import Moveable from "react-moveable";

export default function DraggableText({ block, selected, onSelect, onUpdate, parentRef }) {
  const ref = useRef(null);
  const [frame, setFrame] = useState({ translate: [0, 0] });

  // При выборе блока сбрасываем frame
  useEffect(() => {
    if (selected) setFrame({ translate: [0, 0] });
  }, [selected]);

  return (
    <>
      <div
        ref={ref}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(block.id);
        }}
        style={{
          position: "absolute",
          left: block.x,
          top: block.y,
          fontSize: block.fontSize,
          fontFamily: block.fontFamily,
          color: "#fff",
          cursor: "move",
          background: "transparent",
          border: selected ? "1px dashed #0f0" : "none",
          padding: 2,
          userSelect: "none",
          transform: `translate(${frame.translate[0]}px, ${frame.translate[1]}px)`,
        }}
      >
        {block.text}
      </div>

      {selected && (
        <Moveable
          target={ref.current}
          draggable
          origin={false}
          onDrag={({ beforeTranslate }) => {
            const [dx, dy] = beforeTranslate;
            // Временное смещение (анимация)
            setFrame({ translate: [dx, dy] });
          }}
          onDragEnd={({ lastEvent }) => {
            if (!lastEvent) return;
            const [dx, dy] = lastEvent.beforeTranslate;

            // Обновляем реальные координаты блока (x, y)
            onUpdate(block.id, {
              x: block.x + dx,
              y: block.y + dy,
            });

            // Сбрасываем transform
            setFrame({ translate: [0, 0] });
          }}
        />
      )}
    </>
  );
}
