import React, { useState } from "react";
import WebcamWithText from "./WebcamWithText";
import "./App.css";

function App() {
  const [blocks, setBlocks] = useState([]);
  const [stats, setStats] = useState({
    fps: null,
    avgFps: null,
    modelTime: null,
    fullFrameTime: null,
    modelActive: false
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
          const bgBlock = prev.find(b => b.id === "b1") || { id: "b1" };
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
      </div>

      <div className="webcam-wrapper">
        <WebcamWithText
          blocks={blocks}
          setBlocks={setBlocks}
          selectedBlockId={null}
          setSelectedBlockId={() => {}}
          onStatsUpdate={setStats}
          backgroundImage={blocks[0]?.image || null}
        />
      </div>

      <div className="settings-panel">
        <h3>Настройки</h3>
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
            Изображение:
            <input type="file" accept="image/*" onChange={handleImageChange} />
          </label>
        </div>
        <div className="setting-group">
          <label>
            Информация о сотруднике:
            <input type="file" accept=".json" onChange={handleJsonUpload} />
          </label>
        </div>
      </div>
    </div>
  );
}

export default App;
