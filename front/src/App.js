import React, { useState } from "react";
import WebcamWithText from "./WebcamWithText";

function App() {
  const [blocks, setBlocks] = useState([
    { id: "b1", text: "Привет", x: 50, y: 50, fontSize: 24, fontFamily: "Arial" },
    { id: "b2", text: "Мир", x: 200, y: 100, fontSize: 30, fontFamily: "Courier New" },
  ]);

  const [selectedBlockId, setSelectedBlockId] = useState(null);

  const selectedBlock = blocks.find((b) => b.id === selectedBlockId);

  const handleChange = (key, value) => {
    setBlocks((prev) =>
      prev.map((b) => (b.id === selectedBlockId ? { ...b, [key]: value } : b))
    );
  };

  return (
    <div style={{ display: "flex", gap: "1rem" }}>
      <WebcamWithText
        blocks={blocks}
        setBlocks={setBlocks}
        selectedBlockId={selectedBlockId}
        setSelectedBlockId={setSelectedBlockId}
      />

      {selectedBlock && (
        <div style={{ minWidth: "220px" }}>
          <h3>Редактирование блока</h3>
          <div>
            <label>
              Текст:
              <input
                type="text"
                value={selectedBlock.text}
                onChange={(e) => handleChange("text", e.target.value)}
              />
            </label>
          </div>
          <div>
            <label>
              Размер:
              <input
                type="number"
                value={selectedBlock.fontSize}
                onChange={(e) =>
                  handleChange("fontSize", parseInt(e.target.value) || 1)
                }
              />
            </label>
          </div>
          <div>
            <label>
              Шрифт:
              <select
                value={selectedBlock.fontFamily}
                onChange={(e) => handleChange("fontFamily", e.target.value)}
              >
                <option value="Arial">Arial</option>
                <option value="Courier New">Courier New</option>
                <option value="Times New Roman">Times New Roman</option>
                <option value="Verdana">Verdana</option>
              </select>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
