import React, { useState } from "react";
import { NormalizedField, NormalizedToken } from "../types";
import { ZoomIn, ZoomOut, RotateCcw, Layers } from "lucide-react";

interface DocumentViewerProps {
  imageUrl: string;
  width?: number;
  height?: number;
  tokens?: NormalizedToken[];
  fields?: Record<string, NormalizedField>;
  selectedField?: string | null;
  onSelectField?: (fieldName: string) => void;
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  imageUrl,
  tokens = [],
  fields = {},
  selectedField = null,
  onSelectField,
}) => {
  const [zoom, setZoom] = useState(1);
  const [showTokens, setShowTokens] = useState(false);
  const [hoveredField, setHoveredField] = useState<string | null>(null);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.25, 2.5));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.5));
  const handleReset = () => setZoom(1);

  return (
    <div className="flex flex-col h-full bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
      {/* Controls Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTokens(!showTokens)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition-colors font-medium ${
              showTokens
                ? "bg-indigo-600 text-white"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
            title="Toggle OCR Tokens layer"
          >
            <Layers className="w-3.5 h-3.5" />
            {showTokens ? "Showing All OCR" : "Showing KIE Fields"}
          </button>
        </div>

        <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
          <button
            onClick={handleZoomOut}
            className="p-1 hover:bg-slate-700 rounded text-slate-300 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="px-2 font-mono text-slate-400 select-none text-xs">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-1 hover:bg-slate-700 rounded text-slate-300 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-1 hover:bg-slate-700 rounded text-slate-300 transition-colors ml-1"
            title="Reset Zoom"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className="relative flex-1 overflow-auto p-4 flex items-center justify-center bg-slate-950">
        <div
          className="relative transition-transform duration-150 origin-top shadow-2xl rounded border border-slate-800 max-w-full"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Main Document Image */}
          <img
            src={imageUrl}
            alt="Document Scan"
            className="block max-h-[750px] w-auto object-contain select-none pointer-events-none rounded"
          />

          {/* SVG Overlay using percentage coordinates (0.0 to 1.0 mapped to 0% to 100%) */}
          <svg className="absolute inset-0 w-full h-full pointer-events-auto">
            {/* 1. OCR Tokens (when enabled) */}
            {showTokens &&
              tokens.map((token, idx) => {
                const [x1, y1, x2, y2] = token.bbox;
                const left = `${x1 * 100}%`;
                const top = `${y1 * 100}%`;
                const width = `${Math.max(0.005, x2 - x1) * 100}%`;
                const height = `${Math.max(0.005, y2 - y1) * 100}%`;

                return (
                  <rect
                    key={`token-${idx}`}
                    x={left}
                    y={top}
                    width={width}
                    height={height}
                    fill="rgba(99, 102, 241, 0.08)"
                    stroke="rgba(99, 102, 241, 0.4)"
                    strokeWidth="1"
                    strokeDasharray="2,2"
                    className="hover:fill-indigo-500/25 transition-colors cursor-pointer"
                  >
                    <title>{`${token.text} (${Math.round(token.confidence * 100)}%)`}</title>
                  </rect>
                );
              })}

            {/* 2. KIE Extracted Fields */}
            {Object.entries(fields).map(([name, field]) => {
              if (!field.bbox) return null;
              const [x1, y1, x2, y2] = field.bbox;
              const left = `${x1 * 100}%`;
              const top = `${y1 * 100}%`;
              const width = `${Math.max(0.01, x2 - x1) * 100}%`;
              const height = `${Math.max(0.01, y2 - y1) * 100}%`;

              const isSelected = selectedField === name;
              const isHovered = hoveredField === name;

              let strokeColor = "rgba(16, 185, 129, 0.8)"; // emerald
              let fillColor = "rgba(16, 185, 129, 0.15)";
              if (field.confidence < 0.85) {
                strokeColor = "rgba(245, 158, 11, 0.9)"; // amber
                fillColor = "rgba(245, 158, 11, 0.2)";
              }
              if (isSelected || isHovered) {
                strokeColor = "#6366f1"; // indigo
                fillColor = "rgba(99, 102, 241, 0.35)";
              }

              return (
                <g
                  key={`field-${name}`}
                  className="cursor-pointer transition-all"
                  onClick={() => onSelectField && onSelectField(name)}
                  onMouseEnter={() => setHoveredField(name)}
                  onMouseLeave={() => setHoveredField(null)}
                >
                  <rect
                    x={left}
                    y={top}
                    width={width}
                    height={height}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? "2.5" : "1.5"}
                    rx="3"
                  />
                  {/* Field Label Badge */}
                  <text
                    x={left}
                    y={`calc(${top} - 4px)`}
                    fill="#ffffff"
                    fontSize="10"
                    fontWeight="600"
                    className="select-none font-sans drop-shadow-md"
                  >
                    {name.toUpperCase()}: {field.value}
                  </text>
                  <title>{`${name}: ${field.value} (Confidence: ${Math.round(field.confidence * 100)}%)`}</title>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
};
