"use client";

import React, { useState, useEffect } from "react";
import axios, { AxiosError } from "axios";
import { useSession } from "@/context/SessionContext";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

type Hyperparams = Record<string, string | number | boolean>;

interface TrainingResult {
  status: string;
  problem_type: string;
  target_column: string;
  model_name: string;
  hyperparameters_used: Hyperparams;
  metrics: Record<string, unknown>; // can contain nested objects
  hf_status?: string;
  hf_filename?: string | null;
  huggingface_download_url?: string | null;
  model_path: string;
}

export default function TrainModelNodeCompact() {
  const { sessionId } = useSession(); // ✅ global session
  const [model, setModel] = useState<string>("");
  const [targetCol, setTargetCol] = useState<string>("");
  const [hyperparams, setHyperparams] = useState<Hyperparams>({});
  const [results, setResults] = useState<TrainingResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const modelOptions = [
    "RandomForestClassifier",
    "LogisticRegression",
    "RandomForestRegressor",
    "LinearRegression",
    "KMeans",
  ];

  // Fetch default hyperparameters when model changes
  useEffect(() => {
    if (!model) return;

    axios
      .get<{ default_hyperparameters: Hyperparams }>(
        `http://localhost:7860/hyperparameters?model_name=${model}`
      )
      .then((res) => setHyperparams(res.data.default_hyperparameters))
      .catch(() => setHyperparams({}));
  }, [model]);

  const handleParamChange = (key: string, value: string) => {
    setHyperparams((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleTrain = async () => {
    if (!sessionId) {
      alert("No active session. Please clean/save your data first.");
      return;
    }

    if (!model) {
      alert("Please select a model first!");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("session_id", sessionId); // required by backend
    if (targetCol.trim()) {
      formData.append("target", targetCol.trim()); // optional
    }
    formData.append("model_choice", model);
    formData.append("params", JSON.stringify(hyperparams));

    try {
      const res = await axios.post<TrainingResult>(
        "http://localhost:7860/train-model",
        formData
      );
      setResults(res.data);
      console.log("Training successful:", res.data);
    } catch (err) {
      const error = err as AxiosError<{ detail: string }>;
      alert(error.response?.data?.detail || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  // Only display numeric scalar metrics in the compact card
  const numericMetrics =
    results?.metrics
      ? Object.entries(results.metrics).filter(
          ([, v]) => typeof v === "number" && Number.isFinite(v)
        )
      : [];

  return (
    <div className="w-[280px] bg-[#0b1220] border border-[#253148] rounded-2xl p-3 shadow-md text-gray-200">
      <div className="text-sm font-semibold text-center text-gray-100 mb-2">
        🧠 Model Training
      </div>

      {/* Session ID */}
      <div className="mb-2">
        <Label className="text-[11px] text-gray-400">Session</Label>
        <Input
          value={sessionId || ""}
          readOnly
          className="mt-1 bg-[#0f1724] border-none text-[11px] h-7 px-2 cursor-not-allowed"
        />
      </div>

      {/* Target column (optional) */}
      <div className="mb-2">
        <Label className="text-[11px] text-gray-400">Target (optional)</Label>
        <Input
          value={targetCol}
          onChange={(e) => setTargetCol(e.target.value)}
          placeholder="Leave empty → last column"
          className="mt-1 bg-[#0f1724] border-none text-[11px] h-7 px-2"
        />
      </div>

      {/* Model selector */}
      <div className="mb-2">
        <Label className="text-[11px] text-gray-400">Model</Label>
        <Select onValueChange={setModel}>
          <SelectTrigger className="w-full mt-1 bg-[#0f1724] border-none text-[11px] h-8">
            <SelectValue placeholder="Choose model" />
          </SelectTrigger>
          <SelectContent className="bg-[#0f1724] text-[12px]">
            {modelOptions.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Hyperparameters */}
      {Object.keys(hyperparams).length > 0 && (
        <div className="mb-2">
          <Label className="text-[11px] text-gray-400">Hyperparams</Label>
          <div className="max-h-[110px] overflow-y-auto bg-[#071022] p-2 rounded-md border border-[#1f2a36] text-[11px]">
            {Object.entries(hyperparams).map(([key, val]) => (
              <div key={key} className="flex items-center gap-2 mb-1">
                <div className="w-1/2 text-[11px] text-gray-300 truncate">
                  {key}
                </div>
                <Input
                  value={typeof val === "boolean" ? String(val) : val ?? ""}
                  onChange={(e) => handleParamChange(key, e.target.value)}
                  className="text-[11px] w-1/2 bg-[#071022] border-none h-7 px-2"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Train button */}
      <Button
        onClick={handleTrain}
        disabled={loading}
        className="w-full py-2 text-[13px] font-semibold rounded-md mt-1 bg-gradient-to-r from-[#06b6d4] to-[#3b82f6]"
      >
        {loading ? "Training..." : "🚀 Train"}
      </Button>

      {/* Results */}
      <div className="mt-3 bg-[#071422] border border-[#16232b] rounded-md p-2 text-[12px]">
        {results ? (
          <>
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs font-medium text-gray-100">Results</div>
              <div className="text-[11px] text-gray-400 truncate">
                {results.model_name}
              </div>
            </div>

            <div className="text-[11px] text-gray-300 mb-2">
              <div>
                <strong className="text-gray-200">Problem:</strong>{" "}
                <span className="text-gray-300">{results.problem_type}</span>
              </div>
              <div className="truncate">
                <strong className="text-gray-200">Target:</strong>{" "}
                <span className="text-gray-300">
                  {results.target_column ?? "auto (last column)"}
                </span>
              </div>
              <div className="truncate">
                <strong className="text-gray-200">Path:</strong>{" "}
                <span className="text-gray-300">{results.model_path}</span>
              </div>
              {results.hf_filename && (
                <div className="truncate mt-1">
                  <strong className="text-gray-200">File:</strong>{" "}
                  <span className="text-gray-300">
                    {results.hf_filename}
                  </span>
                </div>
              )}
            </div>

            <div className="text-[11px] text-gray-200 font-medium mb-1">
              Metrics
            </div>
            <div className="max-h-[90px] overflow-y-auto">
              {numericMetrics.length === 0 ? (
                <div className="text-[11px] text-gray-400">
                  No scalar metrics returned
                </div>
              ) : (
                <div className="space-y-1">
                  {numericMetrics.map(([k, v]) => (
                    <div
                      key={k}
                      className="flex items-center justify-between bg-[#06121a] px-2 py-1 rounded"
                    >
                      <div className="text-[11px] text-gray-300 truncate">
                        {k}
                      </div>
                      <div className="text-[11px] text-gray-100 font-semibold">
                        {(v as number).toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Download Model button */}
            {results.huggingface_download_url && (
              <Button
                onClick={() =>
                  window.open(results.huggingface_download_url!, "_blank")
                }
                className="w-full mt-2 py-1 text-[12px] bg-gradient-to-r from-[#1e293b] to-[#334155] hover:opacity-90"
              >
                📥 Download Model
              </Button>
            )}
          </>
        ) : (
          <div className="text-[11px] text-gray-400">No results yet</div>
        )}
      </div>
    </div>
  );
}
