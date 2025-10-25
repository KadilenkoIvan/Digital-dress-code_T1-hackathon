import React, { useState, useEffect, useRef } from "react";

export default function DraggableText({ block, onUpdate, selected, onSelect, parentRef, backgroundLayerRef }) {
  const [dragging, setDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [autoColor, setAutoColor] = useState({ text: block.color, stroke: '#000000', luminance: 0.5, contrast: 4.5 });
  const textRef = useRef(null);
  const dragStateRef = useRef({ 
    isDragging: false, 
    offsetX: 0, 
    offsetY: 0,
    hasMoved: false,
    startX: 0,
    startY: 0
  });

  // Функция для анализа фона и определения оптимального цвета текста
  const analyzeBackgroundAndSetColor = () => {
    if (!backgroundLayerRef?.current || !textRef.current) return;

    try {
      // Создаем временный canvas для анализа фона
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      // Получаем размеры и позицию текста
      const textRect = textRef.current.getBoundingClientRect();
      const bgRect = backgroundLayerRef.current.getBoundingClientRect();
      
      // Вычисляем относительную позицию текста на фоне
      const relX = textRect.left - bgRect.left;
      const relY = textRect.top - bgRect.top;
      const width = Math.min(textRect.width, 200); // Ограничиваем для производительности
      const height = Math.min(textRect.height, 100);
      
      if (width <= 0 || height <= 0) return;
      
      canvas.width = width;
      canvas.height = height;
      
      // Получаем фоновое изображение
      const bgStyle = window.getComputedStyle(backgroundLayerRef.current);
      const bgImage = bgStyle.backgroundImage;
      
      if (bgImage && bgImage !== 'none') {
        // Если есть фоновое изображение, анализируем его
        const img = new Image();
        img.crossOrigin = "anonymous";
        
        // Извлекаем URL из CSS url()
        const match = bgImage.match(/url\(['"]?([^'"]+)['"]?\)/);
        if (match) {
          img.src = match[1];
          img.onload = () => {
            // Рисуем часть фона, соответствующую позиции текста
            const scale = img.naturalWidth / bgRect.width;
            ctx.drawImage(
              img,
              relX * scale, relY * scale,
              width * scale, height * scale,
              0, 0, width, height
            );
            
            // Анализируем цвета
            const imageData = ctx.getImageData(0, 0, width, height);
            const { avgR, avgG, avgB, relativeLuminance } = analyzeImageData(imageData);
            
            // Находим оптимальный контрастный цвет по WCAG AA
            const contrastColor = findContrastColor(avgR, avgG, avgB, 4.5);
            const textColor = `rgb(${contrastColor.r}, ${contrastColor.g}, ${contrastColor.b})`;
            // Обводка — противоположный цвет
            const strokeColor = contrastColor.r > 128 ? '#000000' : '#ffffff';
            
            console.log(`📊 WCAG Analysis: BG(${avgR},${avgG},${avgB}) L=${relativeLuminance.toFixed(3)} → Text: ${textColor} (contrast: ${contrastColor.contrast.toFixed(2)}:1)`);
            
            setAutoColor({ text: textColor, stroke: strokeColor, luminance: relativeLuminance, contrast: contrastColor.contrast });
          };
        }
      } else {
        // Если фон градиентный или однотонный
        const bgColor = bgStyle.backgroundColor;
        const rgb = parseRGB(bgColor);
        if (rgb) {
          const relativeLuminance = getRelativeLuminance(rgb.r, rgb.g, rgb.b);
          
          // Находим оптимальный контрастный цвет по WCAG AA
          const contrastColor = findContrastColor(rgb.r, rgb.g, rgb.b, 4.5);
          const textColor = `rgb(${contrastColor.r}, ${contrastColor.g}, ${contrastColor.b})`;
          const strokeColor = contrastColor.r > 128 ? '#000000' : '#ffffff';
          
          console.log(`📊 WCAG Analysis (gradient): BG(${rgb.r},${rgb.g},${rgb.b}) L=${relativeLuminance.toFixed(3)} → Text: ${textColor} (contrast: ${contrastColor.contrast.toFixed(2)}:1)`);
          
          setAutoColor({ text: textColor, stroke: strokeColor, luminance: relativeLuminance, contrast: contrastColor.contrast });
        }
      }
    } catch (error) {
      console.warn('Background analysis failed:', error);
    }
  };

  // WCAG: Вычисление relative luminance согласно стандарту
  const getRelativeLuminance = (r, g, b) => {
    // Нормализуем значения RGB (0-255) к (0-1)
    const rsRGB = r / 255;
    const gsRGB = g / 255;
    const bsRGB = b / 255;
    
    // Применяем гамма-коррекцию согласно WCAG
    const rLinear = rsRGB <= 0.03928 ? rsRGB / 12.92 : Math.pow((rsRGB + 0.055) / 1.055, 2.4);
    const gLinear = gsRGB <= 0.03928 ? gsRGB / 12.92 : Math.pow((gsRGB + 0.055) / 1.055, 2.4);
    const bLinear = bsRGB <= 0.03928 ? bsRGB / 12.92 : Math.pow((bsRGB + 0.055) / 1.055, 2.4);
    
    // Relative luminance по формуле WCAG
    return 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear;
  };

  // WCAG: Вычисление контрастности между двумя цветами
  const getContrastRatio = (lum1, lum2) => {
    const lighter = Math.max(lum1, lum2);
    const darker = Math.min(lum1, lum2);
    return (lighter + 0.05) / (darker + 0.05);
  };

  // Поиск оптимального контрастного цвета по WCAG AA (4.5:1 для нормального текста)
  const findContrastColor = (bgR, bgG, bgB, targetRatio = 4.5) => {
    const bgLuminance = getRelativeLuminance(bgR, bgG, bgB);
    
    // Вычисляем контрастность с белым и черным
    const whiteLuminance = 1; // белый = 1
    const blackLuminance = 0; // черный = 0
    
    const whiteContrast = getContrastRatio(whiteLuminance, bgLuminance);
    const blackContrast = getContrastRatio(blackLuminance, bgLuminance);
    
    // Выбираем цвет с лучшей контрастностью
    if (whiteContrast > blackContrast) {
      return { r: 255, g: 255, b: 255, contrast: whiteContrast };
    } else {
      return { r: 0, g: 0, b: 0, contrast: blackContrast };
    }
  };

  // Анализ данных изображения для определения средней яркости и доминантного цвета
  const analyzeImageData = (imageData) => {
    const data = imageData.data;
    let totalR = 0, totalG = 0, totalB = 0;
    let pixelCount = 0;
    
    // Анализируем каждый 4-й пиксель для производительности
    for (let i = 0; i < data.length; i += 16) {
      totalR += data[i];
      totalG += data[i + 1];
      totalB += data[i + 2];
      pixelCount++;
    }
    
    const avgR = totalR / pixelCount;
    const avgG = totalG / pixelCount;
    const avgB = totalB / pixelCount;
    
    // Вычисляем relative luminance по WCAG
    const relativeLuminance = getRelativeLuminance(avgR, avgG, avgB);
    
    return {
      avgR: Math.round(avgR),
      avgG: Math.round(avgG),
      avgB: Math.round(avgB),
      relativeLuminance,
      dominantColor: `rgb(${Math.round(avgR)}, ${Math.round(avgG)}, ${Math.round(avgB)})`
    };
  };

  // Парсинг RGB из CSS строки
  const parseRGB = (color) => {
    const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (match) {
      return { r: parseInt(match[1]), g: parseInt(match[2]), b: parseInt(match[3]) };
    }
    return null;
  };

  // Анализируем фон при монтировании и изменении позиции
  useEffect(() => {
    analyzeBackgroundAndSetColor();
  }, [block.x, block.y, block.fontSize, backgroundLayerRef]);

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

  // Определяем финальный цвет текста и обводку
  const finalTextColor = autoColor.text || block.color;
  const strokeColor = autoColor.stroke || '#000000';
  
  // Умная обводка для максимальной читаемости
  const smartTextShadow = `
    -1px -1px 0 ${strokeColor},
    1px -1px 0 ${strokeColor},
    -1px 1px 0 ${strokeColor},
    1px 1px 0 ${strokeColor},
    0 0 3px ${strokeColor},
    0 0 5px ${strokeColor}
  `;

  return (
    <div
      ref={textRef}
      style={{
        position: "absolute",
        left: block.x,
        top: block.y,
        fontSize: `${block.fontSize}px`,
        color: finalTextColor,
        fontFamily: block.fontFamily || 'Arial',
        fontWeight: block.fontWeight || 'normal',
        fontStyle: block.fontStyle || 'normal',
        cursor: cursorStyle,
        userSelect: "none",
        border: selected ? "2px solid #2a5298" : "2px solid transparent",
        padding: "4px 8px",
        borderRadius: "4px",
        backgroundColor: selected ? "rgba(42, 82, 152, 0.1)" : "transparent",
        transition: selected ? "none" : "color 0.3s ease, text-shadow 0.3s ease",
        whiteSpace: "nowrap",
        textShadow: smartTextShadow,
        zIndex: 1, // Текст между фоном (0) и canvas (2)
        // Дополнительная читаемость с помощью webkit
        WebkitFontSmoothing: 'antialiased',
        MozOsxFontSmoothing: 'grayscale'
      }}
      onMouseDown={handleMouseDown}
    >
      {block.text}
    </div>
  );
}