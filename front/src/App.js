import React, { useState } from "react";
import WebcamWithText from "./WebcamWithText";

function App() {
  const [blocks, setBlocks] = useState([]);

  const currentLevel = blocks[0]?.level || "low";

  const handleLevelChange = (level) => {
    if (blocks.length === 0) {
      setBlocks([{ id: "b1", level }]);
    } else {
      setBlocks([{ ...blocks[0], level }]);
    }
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
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const json = JSON.parse(reader.result);
          if (Array.isArray(json)) {
            setBlocks(json);
          } else {
            alert("JSON должен быть массивом блоков!");
          }
        } catch (err) {
          alert("Ошибка при чтении JSON: " + err.message);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div style={{ display: "flex", gap: "1rem" }}>
      <WebcamWithText
        blocks={blocks}
        setBlocks={setBlocks}
        selectedBlockId={null}
        setSelectedBlockId={() => {}}
      />

      <div style={{ minWidth: "220px" }}>
        <h3>Настройки</h3>
        <div>
          <label>
            Выберите уровень приватности:
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
        <div>
          <label>
            Выберите изображение:
            <input type="file" accept="image/*" onChange={handleImageChange} />
          </label>
        </div>
        <div style={{ marginTop: "10px" }}>
          <label>
            Загрузите информацию о сотруднике:
            <input type="file" accept=".json" onChange={handleJsonUpload} />
          </label>
        </div>
      </div>
    </div>
  );
}

export default App;
