import React, { useState, useRef } from "react";
import WebcamWithText from "./WebcamWithText";
import "./App.css";

function App() {
  const [blocks, setBlocks] = useState([]);
  const [selectedBlockId, setSelectedBlockId] = useState(null);
  const [backgroundBlur, setBackgroundBlur] = useState(10); // Размытие фона (0-50)
  const [modelScale, setModelScale] = useState(0.3); // Масштаб модели (0.0-1.0)
  const [downsampleRatio, setDownsampleRatio] = useState(0.9); // Downsample ratio (0.5-0.9)
  const [rawMode, setRawMode] = useState(false); // Режим вывода: false = обработанное, true = сырое видео
  const [numThreads, setNumThreads] = useState(() => {
    // Восстанавливаем значение из localStorage после перезагрузки
    const saved = localStorage.getItem('onnx_num_threads');
    return saved ? parseInt(saved) : 1;
  }); // Количество потоков
  const [stats, setStats] = useState({
    fps: null,
    avgFps: null,
    modelTime: null,
    fullFrameTime: null,
    modelActive: false,
    device: 'Loading...'
  });
  
  // Refs для сброса file inputs
  const backgroundInputRef = useRef(null);
  const jsonInputRef = useRef(null);

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

  const handleRemoveBackground = () => {
    setBlocks(prev => {
      if (prev.length === 0) return prev;
      const bgBlock = prev[0];
      if (bgBlock.id === "b1") {
        const { image, ...rest } = bgBlock;
        return [rest, ...prev.slice(1)];
      }
      return prev;
    });
    // Сбрасываем input чтобы можно было загрузить тот же файл снова
    if (backgroundInputRef.current) {
      backgroundInputRef.current.value = '';
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

  const handleRemoveEmployee = () => {
    // Удаляем все текстовые блоки, созданные из JSON, и информацию о сотруднике
    setBlocks(prev => {
      if (prev.length === 0) return prev;
      const bgBlock = prev[0];
      if (bgBlock.id === "b1") {
        const { employee, ...rest } = bgBlock;
        // Оставляем только блок метаданных (без employee) и пользовательские блоки (не из JSON)
        return [rest];
      }
      return prev;
    });
    // Сбрасываем input чтобы можно было загрузить тот же файл снова
    if (jsonInputRef.current) {
      jsonInputRef.current.value = '';
    }
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
          <span className="stat-label">Время кадра:</span>
          <span className="stat-value">{stats.fullFrameTime !== null ? `${stats.fullFrameTime} ms` : 'None'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Модель:</span>
          <span className={`stat-value ${stats.modelActive ? 'active' : 'inactive'}`}>
            {stats.modelActive ? 'Активна' : 'Неактивна'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Device:</span>
          <span className="stat-value">{stats.device}</span>
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
          rawMode={rawMode}
          numThreads={numThreads}
        />
      </div>

      <div className="settings-panel">
        <h3>Настройки</h3>
        
        {/* Тумблер режима вывода */}
        <div className="setting-group">
          <label>
            Режим отображения:
            <div className="toggle-switch">
              <input 
                type="checkbox" 
                id="raw-mode-toggle"
                checked={rawMode}
                onChange={(e) => setRawMode(e.target.checked)}
              />
              <label htmlFor="raw-mode-toggle" className="toggle-slider">
                <span className="toggle-option toggle-processed"> Обработанное</span>
                <span className="toggle-option toggle-raw"> Сырое видео</span>
              </label>
            </div>
          </label>
        </div>

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
            Количество потоков CPU: {numThreads}
            <input 
              type="range" 
              min="1" 
              max={navigator.hardwareConcurrency || 4} 
              step="1"
              value={numThreads} 
              onChange={(e) => setNumThreads(parseInt(e.target.value))}
              className="threads-slider"
            />
            <span className="slider-hint">Сколько потоков процессора использовать</span>
            <span className="slider-hint">(Внимание! Перезагружает страницу при изменении)</span>
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
            <div className="file-input-with-button">
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleImageChange}
                ref={backgroundInputRef}
              />
              {blocks[0]?.image && (
                <button 
                  className="remove-btn"
                  onClick={handleRemoveBackground}
                  title="Удалить фон"
                >
                  🗑️ Удалить
                </button>
              )}
            </div>
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
            <div className="file-input-with-button">
              <input 
                type="file" 
                accept=".json" 
                onChange={handleJsonUpload}
                ref={jsonInputRef}
              />
              {blocks[0]?.employee && (
                <button 
                  className="remove-btn"
                  onClick={handleRemoveEmployee}
                  title="Удалить информацию о сотруднике"
                >
                  🗑️ Удалить
                </button>
              )}
            </div>
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
