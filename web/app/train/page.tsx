"use client";

import React, { useState, useEffect } from "react";
import axios, { AxiosError } from "axios";
import { useSession } from "@/context/SessionContext";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  metrics: Record<string, any>;
  hf_status?: string;
  hf_filename?: string | null;
  huggingface_download_url?: string | null;
  model_path: string;
}

export default function TrainModelPage() {
  const { sessionId } = useSession(); // 🔥 Global session
  const [targetCol, setTargetCol] = useState<string>("");
  const [model, setModel] = useState<string>("");
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

  // Fetch default hyperparameters when user selects model
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
      alert("⚠️ No active session. Please clean and save your dataset first!");
      return;
    }

    if (!model) {
      alert("Please select a model first!");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("session_id", sessionId);
    if (targetCol.trim()) formData.append("target", targetCol.trim());
    formData.append("model_choice", model);
    formData.append("params", JSON.stringify(hyperparams));

    try {
      const res = await axios.post<TrainingResult>(
        "http://localhost:7860/train-model",
        formData
      );
      setResults(res.data);
    } catch (err) {
      const error = err as AxiosError<{ detail: string }>;
      alert(error.response?.data?.detail || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f9fafb] dark:bg-gray-900 text-gray-900 dark:text-white p-6">
      <Card className="max-w-3xl mx-auto shadow-lg rounded-2xl border border-gray-200 dark:border-gray-700">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center">
            🧠 Train Machine Learning Model
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Session ID (read-only) */}
          <div>
            <Label className="text-sm font-medium">Active Session ID</Label>
            <Input
              value={sessionId || ""}
              readOnly
              className="mt-2 bg-gray-200 dark:bg-gray-800 cursor-not-allowed"
            />
            <p className="text-xs mt-1 text-gray-500">
              This identifies your cleaned dataset on Hugging Face.
            </p>
          </div>

          {/* Target column (optional) */}
          <div>
            <Label className="text-sm font-medium">
              Target Column (optional)
            </Label>
            <Input
              placeholder="Leave empty to use the last column"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="mt-2"
            />
          </div>

          {/* Model Selector */}
          <div>
            <Label className="text-sm font-medium">Select Model</Label>
            <Select onValueChange={setModel}>
              <SelectTrigger className="w-full mt-2">
                <SelectValue placeholder="Pick a model" />
              </SelectTrigger>
              <SelectContent>
                {modelOptions.map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Hyperparameters UI */}
          {Object.keys(hyperparams).length > 0 && (
            <div className="space-y-3">
              <Label className="text-sm font-medium">Hyperparameters</Label>
              {Object.entries(hyperparams).map(([key, val]) => (
                <div key={key} className="flex items-center gap-3">
                  <Label className="w-1/3">{key}</Label>
                  <Input
                    value={String(val)}
                    onChange={(e) => handleParamChange(key, e.target.value)}
                    className="w-2/3"
                  />
                </div>
              ))}
            </div>
          )}

          {/* Train Button */}
          <Button
            onClick={handleTrain}
            disabled={loading}
            className="w-full mt-4"
          >
            {loading ? "⏳ Training..." : "🚀 Train Model"}
          </Button>

          {/* Results */}
          {results && (
            <div className="mt-6 bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
              <h2 className="text-lg font-semibold mb-2">📊 Results</h2>

              <p><strong>Model:</strong> {results.model_name}</p>
              <p><strong>Type:</strong> {results.problem_type}</p>

              <pre className="text-xs bg-gray-200 dark:bg-gray-900 p-2 rounded mt-2 whitespace-pre-wrap">
                {JSON.stringify(results.metrics, null, 2)}
              </pre>

              {results.huggingface_download_url && (
                <p className="mt-2 text-sm">
                  <strong>📥 Download Model:</strong>{" "}
                  <a
                    href={results.huggingface_download_url}
                    className="text-blue-500 underline"
                    target="_blank"
                  >
                    HuggingFace File
                  </a>
                </p>
              )}

              {results.model_path && (
                <p className="text-xs text-gray-500 mt-1">
                  Server file: {results.model_path}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
