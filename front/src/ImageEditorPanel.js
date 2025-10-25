import React, { useState, useEffect, useCallback } from "react";
import "./TextEditorPanel.css"; // Используем те же стили

export default function ImageEditorPanel({ block, onUpdate, onClose }) {
  const [width, setWidth] = useState(block.width || 200);
  const [height, setHeight] = useState(block.height || 200);
  const [objectFit, setObjectFit] = useState(block.objectFit || 'contain');
  const [maintainAspectRatio, setMaintainAspectRatio] = useState(true);
  const [aspectRatio, setAspectRatio] = useState(1);

  // Синхронизация только при смене блока
  useEffect(() => {
    setWidth(block.width || 200);
    setHeight(block.height || 200);
    setObjectFit(block.objectFit || 'contain');
    
    // Вычисляем соотношение сторон
    const ratio = (block.width || 200) / (block.height || 200);
    setAspectRatio(ratio);
  }, [block.id]);

  const handleDelete = () => {
    if (window.confirm('Удалить это изображение?')) {
      onClose(block.id, true); // true = delete
    }
  };

  // Применяем изменения в реальном времени
  const applyChanges = useCallback(() => {
    onUpdate(block.id, {
      width: parseInt(width),
      height: parseInt(height),
      objectFit
    });
  }, [block.id, width, height, objectFit, onUpdate]);

  useEffect(() => {
    applyChanges();
  }, [applyChanges]);

  const handleWidthChange = (newWidth) => {
    setWidth(newWidth);
    if (maintainAspectRatio) {
      setHeight(Math.round(newWidth / aspectRatio));
    }
  };

  const handleHeightChange = (newHeight) => {
    setHeight(newHeight);
    if (maintainAspectRatio) {
      setWidth(Math.round(newHeight * aspectRatio));
    }
  };

  return (
    <div className="text-editor-panel" onClick={(e) => e.stopPropagation()}>
      <div className="text-editor-header">
        <h3>Редактор изображения</h3>
        <button className="close-btn" onClick={() => onClose(null, false)}>✕</button>
      </div>

      <div className="text-editor-content">
        <div className="editor-group">
          <label>
            <input
              type="checkbox"
              checked={maintainAspectRatio}
              onChange={(e) => setMaintainAspectRatio(e.target.checked)}
            />
            {' '}Сохранять пропорции
          </label>
        </div>

        <div className="editor-row">
          <div className="editor-group">
            <label>Ширина (px):</label>
            <input
              type="number"
              value={width}
              onChange={(e) => handleWidthChange(e.target.value)}
              min="50"
              max="1280"
            />
          </div>

          <div className="editor-group">
            <label>Высота (px):</label>
            <input
              type="number"
              value={height}
              onChange={(e) => handleHeightChange(e.target.value)}
              min="50"
              max="960"
            />
          </div>
        </div>

        <div className="editor-group">
          <label>Режим масштабирования:</label>
          <select value={objectFit} onChange={(e) => setObjectFit(e.target.value)}>
            <option value="contain">Вписать (Contain)</option>
            <option value="cover">Заполнить (Cover)</option>
            <option value="fill">Растянуть (Fill)</option>
            <option value="scale-down">Уменьшить (Scale Down)</option>
          </select>
        </div>

        <div className="editor-preview">
          <label>Предпросмотр:</label>
          <div 
            style={{
              width: '100%',
              height: '200px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#1a1a2e',
              borderRadius: '6px',
              overflow: 'hidden'
            }}
          >
            <img
              src={block.imageSrc}
              alt="preview"
              style={{
                width: `${Math.min(width, 300)}px`,
                height: `${Math.min(height, 180)}px`,
                objectFit: objectFit
              }}
            />
          </div>
        </div>

        <div className="editor-actions">
          <button className="delete-btn" onClick={handleDelete}>
            🗑️ Удалить
          </button>
          <button className="close-panel-btn" onClick={() => onClose(null, false)}>
            Закрыть
          </button>
        </div>
      </div>
    </div>
  );
}

