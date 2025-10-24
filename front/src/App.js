import React, { useState } from "react";
import WebcamWithText from "./WebcamWithText";
import "./App.css";

function App() {
  const [blocks, setBlocks] = useState([]);
  const [selectedBlockId, setSelectedBlockId] = useState(null);
  const [backgroundBlur, setBackgroundBlur] = useState(0); // Размытие фона (0-50)
  const [modelScale, setModelScale] = useState(0.2); // Масштаб модели (0.0-1.0)
  const [downsampleRatio, setDownsampleRatio] = useState(0.8); // Downsample ratio (0.5-0.9)
  const [stats, setStats] = useState({
    fps: null,
    avgFps: null,
    modelTime: null,
    fullFrameTime: null,
    modelActive: false,
    backend: 'Loading...'
  });

  const currentLevel = blocks[0]?.level || "low";

  const handleLevelChange = (level) => {
    if (blocks.length === 0) return;
    setBlocks(prev => prev.map(b => b.id === "b1" ? { ...b, level } : b));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        if (blocks.length === 0) {
          setBlocks([{ id: "b1", image: reader.result, level: currentLevel }]);
        } else {
          setBlocks([{ ...blocks[0], image: reader.result }]);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleJsonUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const json = JSON.parse(reader.result);
        if (!json.employee) {
          alert("JSON должен содержать объект employee!");
          return;
        }

        setBlocks(prev => {
          const bgBlock = prev.find(b => b.id === "b1") || { id: "b1", type: "metadata" };
          return [{
            ...bgBlock,
            employee: json.employee,
            level: json.employee.privacy_level || "low"
          }];
        });
      } catch (err) {
        alert("Ошибка при чтении JSON: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  const handleAddTextBlock = () => {
    const newId = `text_${Date.now()}`;
    const newBlock = {
      id: newId,
      type: 'text', // Явно указываем тип
      text: "Новый текст",
      x: 50,
      y: 300,
      fontSize: 20,
      color: "white",
      fontFamily: "Arial",
      fontWeight: "normal",
      fontStyle: "normal"
    };
    setBlocks(prev => [...prev, newBlock]);
    setSelectedBlockId(newId);
  };

  const handleAddImageBlock = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Проверяем тип файла
    if (!file.type.startsWith('image/')) {
      alert('Пожалуйста, выберите изображение!');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const newId = `image_${Date.now()}`;
      const newBlock = {
        id: newId,
        type: 'image',
        imageSrc: reader.result,
        x: 100,
        y: 100,
        width: 200,
        height: 200,
        objectFit: 'contain'
      };
      setBlocks(prev => [...prev, newBlock]);
      setSelectedBlockId(newId);
    };
    reader.readAsDataURL(file);
    
    // Сбрасываем input чтобы можно было загрузить тот же файл снова
    e.target.value = '';
  };

  return (
    <div className="app-container">
      <div className="stats-panel">
        <h3>Статистика</h3>
        <div className="stat-item">
          <span className="stat-label">FPS:</span>
          <span className="stat-value">{stats.fps !== null ? stats.fps : 'None'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Средний FPS:</span>
          <span className="stat-value">{stats.avgFps !== null ? stats.avgFps : 'None'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Время модели:</span>
          <span className="stat-value">{stats.modelTime !== null ? `${stats.modelTime} ms` : 'None'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Время кадра (полное):</span>
          <span className="stat-value">{stats.fullFrameTime !== null ? `${stats.fullFrameTime} ms` : 'None'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Модель:</span>
          <span className={`stat-value ${stats.modelActive ? 'active' : 'inactive'}`}>
            {stats.modelActive ? 'Активна' : 'Неактивна'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Backend:</span>
          <span className="stat-value">{stats.backend}</span>
        </div>
      </div>

      <div className="webcam-wrapper">
        <WebcamWithText
          blocks={blocks}
          setBlocks={setBlocks}
          selectedBlockId={selectedBlockId}
          setSelectedBlockId={setSelectedBlockId}
          onStatsUpdate={setStats}
          backgroundImage={blocks[0]?.image || null}
          backgroundBlur={backgroundBlur}
          modelScale={modelScale}
          downsampleRatio={downsampleRatio}
        />
      </div>

      <div className="settings-panel">
        <h3>Настройки</h3>
        <div className="setting-group">
          <label>
            Масштаб модели (качество): {modelScale.toFixed(2)}
            <input 
              type="range" 
              min="0.1" 
              max="1.0" 
              step="0.05"
              value={modelScale} 
              onChange={(e) => setModelScale(parseFloat(e.target.value))}
              className="model-scale-slider"
            />
            <span className="slider-hint">Меньше = быстрее, но ниже качество</span>
          </label>
        </div>
        <div className="setting-group">
          <label>
            Downsample Ratio: {downsampleRatio.toFixed(2)}
            <input 
              type="range" 
              min="0.5" 
              max="1.0" 
              step="0.05"
              value={downsampleRatio} 
              onChange={(e) => setDownsampleRatio(parseFloat(e.target.value))}
              className="downsample-slider"
            />
            <span className="slider-hint">Меньше = быстрее, больше = лучше качество краёв</span>
          </label>
        </div>
        <div className="setting-group">
          <label>
            Уровень отображения персональных данных:
            <select
              value={currentLevel}
              onChange={(e) => handleLevelChange(e.target.value)}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </label>
        </div>
        <div className="setting-group">
          <label>
            Задний фон:
            <input type="file" accept="image/*" onChange={handleImageChange} />
          </label>
        </div>
        <div className="setting-group">
          <label>
            Размытие фона: {backgroundBlur}px
            <input 
              type="range" 
              min="0" 
              max="50" 
              step="1"
              value={backgroundBlur} 
              onChange={(e) => setBackgroundBlur(parseFloat(e.target.value))}
              className="blur-slider"
            />
          </label>
        </div>
        <div className="setting-group">
          <label>
            Информация о сотруднике:
            <input type="file" accept=".json" onChange={handleJsonUpload} />
          </label>
        </div>
        <div className="setting-group">
          <button 
            className="add-text-btn"
            onClick={handleAddTextBlock}
          >
            ➕ Добавить текст
          </button>
        </div>
        <div className="setting-group">
          <label className="add-image-label">
            🖼️ Добавить картинку
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleAddImageBlock}
              style={{ display: 'none' }}
            />
          </label>
        </div>
      </div>
    </div>
  );
}

export default App;
