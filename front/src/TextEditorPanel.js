import React, { useState, useEffect, useCallback } from "react";
import "./TextEditorPanel.css";

export default function TextEditorPanel({ block, onUpdate, onClose }) {
  const [text, setText] = useState(block.text);
  const [fontSize, setFontSize] = useState(block.fontSize);
  const [color, setColor] = useState(block.color);
  const [fontFamily, setFontFamily] = useState(block.fontFamily || 'Arial');
  const [fontWeight, setFontWeight] = useState(block.fontWeight || 'normal');
  const [fontStyle, setFontStyle] = useState(block.fontStyle || 'normal');

  // Синхронизация только при смене блока (при открытии панели для другого блока)
  useEffect(() => {
    setText(block.text);
    setFontSize(block.fontSize);
    setColor(block.color);
    setFontFamily(block.fontFamily || 'Arial');
    setFontWeight(block.fontWeight || 'normal');
    setFontStyle(block.fontStyle || 'normal');
  }, [block.id]); // Только при смене ID блока

  const handleDelete = () => {
    if (window.confirm('Удалить этот текстовый блок?')) {
      onClose(block.id, true); // true = delete
    }
  };

  // Применяем изменения в реальном времени с использованием useCallback
  const applyChanges = useCallback(() => {
    onUpdate(block.id, {
      text,
      fontSize: parseInt(fontSize),
      color,
      fontFamily,
      fontWeight,
      fontStyle
    });
  }, [block.id, text, fontSize, color, fontFamily, fontWeight, fontStyle, onUpdate]);

  useEffect(() => {
    applyChanges();
  }, [applyChanges]);

  const fonts = [
    'Arial',
    'Helvetica',
    'Times New Roman',
    'Georgia',
    'Courier New',
    'Verdana',
    'Trebuchet MS',
    'Comic Sans MS',
    'Impact',
    'Lucida Console',
    'Segoe UI'
  ];

  return (
    <div className="text-editor-panel" onClick={(e) => e.stopPropagation()}>
      <div className="text-editor-header">
        <h3>Редактор текста</h3>
        <button className="close-btn" onClick={() => onClose(null, false)}>✕</button>
      </div>

      <div className="text-editor-content">
        <div className="editor-group">
          <label>Текст:</label>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Введите текст"
          />
        </div>

        <div className="editor-group">
          <label>Шрифт:</label>
          <select value={fontFamily} onChange={(e) => setFontFamily(e.target.value)}>
            {fonts.map(font => (
              <option key={font} value={font} style={{ fontFamily: font }}>
                {font}
              </option>
            ))}
          </select>
        </div>

        <div className="editor-row">
          <div className="editor-group">
            <label>Размер:</label>
            <input
              type="number"
              value={fontSize}
              onChange={(e) => setFontSize(e.target.value)}
              min="8"
              max="120"
            />
          </div>

          <div className="editor-group">
            <label>Цвет:</label>
            <input
              type="color"
              value={color.startsWith('rgb') ? '#ffffff' : color}
              onChange={(e) => setColor(e.target.value)}
            />
          </div>
        </div>

        <div className="editor-row">
          <div className="editor-group">
            <label>Начертание:</label>
            <select value={fontWeight} onChange={(e) => setFontWeight(e.target.value)}>
              <option value="normal">Обычный</option>
              <option value="bold">Жирный</option>
              <option value="lighter">Тонкий</option>
            </select>
          </div>

          <div className="editor-group">
            <label>Стиль:</label>
            <select value={fontStyle} onChange={(e) => setFontStyle(e.target.value)}>
              <option value="normal">Обычный</option>
              <option value="italic">Курсив</option>
            </select>
          </div>
        </div>

        <div className="editor-preview">
          <label>Предпросмотр:</label>
          <div 
            className="preview-text"
            style={{
              fontSize: `${fontSize}px`,
              color: color,
              fontFamily: fontFamily,
              fontWeight: fontWeight,
              fontStyle: fontStyle
            }}
          >
            {text || 'Пример текста'}
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

