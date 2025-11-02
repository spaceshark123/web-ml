"use client"

import { useState, useEffect } from "react"
import { ArrowLeft, TrendingUp, TrendingDown, Minus } from "lucide-react"
import { Button } from "./ui/button"
import { Card, CardHeader, CardTitle, CardDescription } from "./ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Link } from "react-router-dom"
import { API_BASE_URL } from "@/constants"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"

interface Model {
  id: number
  name: string
  model_type: string
  dataset_id?: number
  metrics: Record<string, any>
}

interface DetailedModel extends Model {
  cv?: Record<string, number>
}

interface ComparisonData {
  model1: DetailedModel
  model2: DetailedModel
}

export function CompareModels() {
  const [models, setModels] = useState<Model[]>([])
  const [datasets, setDatasets] = useState<Array<{ id: number; name: string }>>([])
  const [selectedDataset, setSelectedDataset] = useState<string>("all")
  const [selectedModel1, setSelectedModel1] = useState<number | null>(null)
  const [selectedModel2, setSelectedModel2] = useState<number | null>(null)
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null)
  const [loading, setLoading] = useState(false)
  const [sortBy, setSortBy] = useState<string>("created_at")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")

  useEffect(() => {
    fetchModels()
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`, {
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      })
      if (response.ok) {
        const data = await response.json()
        setDatasets(data.map((ds: any) => ({ id: ds.id, name: ds.name })))
      }
    } catch (error) {
      console.error("Failed to fetch datasets:", error)
    }
  }

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models/compare`, {
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      })
      if (response.ok) {
        const data = await response.json()
        setModels(data)
      }
    } catch (error) {
      console.error("Failed to fetch models:", error)
    }
  }

  const handleCompare = async () => {
    if (!selectedModel1 || !selectedModel2 || selectedModel1 === selectedModel2) {
      alert("Please select two different models")
      return
    }

    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/models/${selectedModel1}/compare/${selectedModel2}`, {
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      })
      if (response.ok) {
        const data = await response.json()
        setComparisonData(data)
        fetchModels()  // Refresh models list to get updated metrics
      }
    } catch (error) {
      console.error("Failed to compare models:", error)
    } finally {
      setLoading(false)
    }
  }

  const sortedModels = [...models].sort((a, b) => {
    const aVal = a.metrics[sortBy] ?? 0
    const bVal = b.metrics[sortBy] ?? 0
    console.log(a.metrics, b.metrics)
    const result = aVal > bVal ? 1 : -1
    return sortOrder === "asc" ? result : -result
  })

  const filteredModels = sortedModels.filter((model) => {
    if (selectedDataset === "all") return true
    return model.dataset_id === Number.parseInt(selectedDataset)
  })

  const getMetricComparison = (_: string, val1: any, val2: any) => {
    if (val1 === null || val2 === null || val1 === undefined || val2 === undefined) {
      return null
    }

    const diff = val1 - val2
    const percentDiff = ((diff / val2) * 100).toFixed(2)

    if (diff > 0) {
      return {
        icon: <TrendingUp className="w-4 h-4" />,
        color: "text-green-600",
        text: `+${percentDiff}%`,
      }
    } else if (diff < 0) {
      return {
        icon: <TrendingDown className="w-4 h-4" />,
        color: "text-red-600",
        text: `${percentDiff}%`,
      }
    } else {
      return {
        icon: <Minus className="w-4 h-4" />,
        color: "text-gray-600",
        text: "0%",
      }
    }
  }

  const renderMetric = (label: string, val1: any, val2: any) => {
    const comparison = getMetricComparison(label, val1, val2)

    return (
      <div key={label} className="grid grid-cols-3 gap-4 py-3 border-b border-gray-200 last:border-0">
        <div className="text-sm font-medium text-gray-700">{label}</div>
        <div className="text-sm text-gray-600">{typeof val1 === "number" ? val1.toFixed(4) : val1 || "N/A"}</div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">{typeof val2 === "number" ? val2.toFixed(4) : val2 || "N/A"}</span>
          {comparison && <div className={`flex items-center gap-1 ${comparison.color}`}>{comparison.icon}</div>}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Back button */}
      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <Link to="/dashboard" className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 text-sm">
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </Link>
      </div>

      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Compare Models</h1>
        <p className="text-gray-500">Side-by-side model comparison and performance metrics</p>
      </div>

  <div className="p-8 grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Model Selection */}
        <div className="lg:col-span-1">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Select Models</h3>

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">Filter by Dataset</label>
                <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                  <SelectTrigger className="bg-white border-gray-300">
                    <SelectValue placeholder="All Datasets" />
                  </SelectTrigger>
                  <SelectContent className="bg-white">
                    <SelectItem value="all">All Datasets</SelectItem>
                    {datasets.map((ds) => (
                      <SelectItem key={ds.id} value={ds.id.toString()}>
                        {ds.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">Model 1</label>
                <Select
                  value={selectedModel1?.toString() || ""}
                  onValueChange={(v) => setSelectedModel1(Number.parseInt(v))}
                >
                  <SelectTrigger className="bg-white border-gray-300">
                    <SelectValue placeholder="Select first model" />
                  </SelectTrigger>
                  <SelectContent className="bg-white">
                    {filteredModels.map((m) => (
                      <SelectItem key={m.id} value={m.id.toString()}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">Model 2</label>
                <Select
                  value={selectedModel2?.toString() || ""}
                  onValueChange={(v) => setSelectedModel2(Number.parseInt(v))}
                >
                  <SelectTrigger className="bg-white border-gray-300">
                    <SelectValue placeholder="Select second model" />
                  </SelectTrigger>
                  <SelectContent className="bg-white">
                    {filteredModels.map((m) => (
                      <SelectItem key={m.id} value={m.id.toString()}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Button
                onClick={handleCompare}
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
              >
                {loading ? "Comparing..." : "Compare"}
              </Button>
            </div>
          </Card>
        </div>

        {/* Comparison Results */}
        <div className="lg:col-span-3">
          {comparisonData ? (
            <div className="space-y-6">
              {/* Model Info Cards */}
              <div className="grid grid-cols-2 gap-6">
                <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                  <CardHeader>
                    <CardTitle className="text-xl">{comparisonData.model1.name}</CardTitle>
                    <CardDescription>{comparisonData.model1.model_type}</CardDescription>
                  </CardHeader>
                </Card>

                <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                  <CardHeader>
                    <CardTitle className="text-xl">{comparisonData.model2.name}</CardTitle>
                    <CardDescription>{comparisonData.model2.model_type}</CardDescription>
                  </CardHeader>
                </Card>
              </div>

              {/* Metrics Comparison */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>

                {/* Classification Metrics */}
                {(comparisonData.model1.metrics.accuracy !== undefined ||
                  comparisonData.model2.metrics.accuracy !== undefined) && (
                  <div>
                    <h4 className="font-medium text-gray-700 mb-3">Classification Metrics</h4>
                    {renderMetric(
                      "Accuracy",
                      comparisonData.model1.metrics.accuracy,
                      comparisonData.model2.metrics.accuracy,
                    )}
                    {renderMetric(
                      "Precision",
                      comparisonData.model1.metrics.precision,
                      comparisonData.model2.metrics.precision,
                    )}
                    {renderMetric("Recall", comparisonData.model1.metrics.recall, comparisonData.model2.metrics.recall)}
                    {renderMetric("F1 Score", comparisonData.model1.metrics.f1, comparisonData.model2.metrics.f1)}
                    {renderMetric(
                      "ROC-AUC",
                      comparisonData.model1.metrics.roc_auc,
                      comparisonData.model2.metrics.roc_auc,
                    )}
                    {(comparisonData.model1.metrics.pr_auc !== undefined || comparisonData.model2.metrics.pr_auc !== undefined) && (
                      renderMetric(
                        "PR-AUC",
                        comparisonData.model1.metrics.pr_auc,
                        comparisonData.model2.metrics.pr_auc,
                      )
                    )}
                  </div>
                )}

                {/* Regression Metrics */}
                {(comparisonData.model1.metrics.mse !== undefined ||
                  comparisonData.model2.metrics.mse !== undefined) && (
                  <div className="mt-6 pt-6 border-t border-gray-200">
                    <h4 className="font-medium text-gray-700 mb-3">Regression Metrics</h4>
                    {renderMetric("MSE", comparisonData.model1.metrics.mse, comparisonData.model2.metrics.mse)}
                    {renderMetric("MAE", comparisonData.model1.metrics.mae, comparisonData.model2.metrics.mae)}
                    {renderMetric("R² Score", comparisonData.model1.metrics.r2, comparisonData.model2.metrics.r2)}
                  </div>
                )}

                {/* Preprocessing Info */}
                {comparisonData.model1.metrics.preprocessing && (
                  <div className="mt-6 pt-6 border-t border-gray-200">
                    <h4 className="font-medium text-gray-700 mb-3">Preprocessing (Model 1)</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Original Rows:</span>
                        <span className="ml-2 font-medium">
                          {comparisonData.model1.metrics.preprocessing.original_rows}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Final Rows:</span>
                        <span className="ml-2 font-medium">
                          {comparisonData.model1.metrics.preprocessing.final_rows}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Missing Removed:</span>
                        <span className="ml-2 font-medium">
                          {comparisonData.model1.metrics.preprocessing.missing_values_removed}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Duplicates Removed:</span>
                        <span className="ml-2 font-medium">
                          {comparisonData.model1.metrics.preprocessing.duplicates_removed}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Class Imbalance */}
                {comparisonData.model1.metrics.preprocessing?.imbalance && (
                  <div className="mt-6 pt-6 border-t border-gray-200 bg-yellow-50 p-4 rounded">
                    <h4 className="font-medium text-gray-700 mb-2">Class Imbalance Detected</h4>
                    <div className="text-sm space-y-1">
                      <p>
                        <span className="text-gray-600">Imbalance Ratio:</span>
                        <span className="ml-2 font-medium">
                          {(comparisonData.model1.metrics.preprocessing.imbalance.imbalance_ratio * 100).toFixed(1)}%
                        </span>
                      </p>
                      <p>
                        <span className="text-gray-600">Minority Class:</span>
                        <span className="ml-2 font-medium">
                          {comparisonData.model1.metrics.preprocessing.imbalance.minority_class_percentage.toFixed(1)}%
                        </span>
                      </p>
                    </div>
                  </div>
                )}
              </Card>

              {/* Curves */}
              {(comparisonData.model1.metrics.roc_curve || comparisonData.model2.metrics.roc_curve) && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">ROC Curve</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="fpr" name="FPR" domain={[0, 1]} />
                      <YAxis type="number" dataKey="tpr" name="TPR" domain={[0, 1]} />
                      <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)} />
                      <Legend />
                      {comparisonData.model1.metrics.roc_curve && (
                        <Line data={comparisonData.model1.metrics.roc_curve} type="monotone" dataKey="tpr" name={`${comparisonData.model1.name}`} stroke="#2563eb" dot={false} />
                      )}
                      {comparisonData.model2.metrics.roc_curve && (
                        <Line data={comparisonData.model2.metrics.roc_curve} type="monotone" dataKey="tpr" name={`${comparisonData.model2.name}`} stroke="#7c3aed" dot={false} />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              )}

              {((comparisonData.model1.metrics.preprocessing?.imbalance?.is_imbalanced && comparisonData.model1.metrics.pr_curve) ||
                (comparisonData.model2.metrics.preprocessing?.imbalance?.is_imbalanced && comparisonData.model2.metrics.pr_curve)) && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Precision-Recall Curve</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="recall" name="Recall" domain={[0, 1]} />
                      <YAxis type="number" dataKey="precision" name="Precision" domain={[0, 1]} />
                      <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)} />
                      <Legend />
                      {comparisonData.model1.metrics.pr_curve && (
                        <Line data={comparisonData.model1.metrics.pr_curve} type="monotone" dataKey="precision" name={`${comparisonData.model1.name}`} stroke="#16a34a" dot={false} />
                      )}
                      {comparisonData.model2.metrics.pr_curve && (
                        <Line data={comparisonData.model2.metrics.pr_curve} type="monotone" dataKey="precision" name={`${comparisonData.model2.name}`} stroke="#ea580c" dot={false} />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              )}

              {/* Cross-validation summary */}
              {(comparisonData.model1.cv || comparisonData.model2.cv) && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Cross-Validation (3-fold)</h3>
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium text-gray-700 mb-2">{comparisonData.model1.name}</h4>
                      <div className="space-y-1 text-sm text-gray-700">
                        {comparisonData.model1.cv ? (
                          Object.entries(comparisonData.model1.cv).map(([k, v]) => {
                            const val = v as number
                            return (
                              <div key={k} className="flex justify-between"><span>{k}</span><span className="font-mono">{typeof val === 'number' ? val.toFixed(4) : String(val)}</span></div>
                            )
                          })
                        ) : (
                          <div className="text-gray-500">CV not available</div>
                        )}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-700 mb-2">{comparisonData.model2.name}</h4>
                      <div className="space-y-1 text-sm text-gray-700">
                        {comparisonData.model2.cv ? (
                          Object.entries(comparisonData.model2.cv).map(([k, v]) => {
                            const val = v as number
                            return (
                              <div key={k} className="flex justify-between"><span>{k}</span><span className="font-mono">{typeof val === 'number' ? val.toFixed(4) : String(val)}</span></div>
                            )
                          })
                        ) : (
                          <div className="text-gray-500">CV not available</div>
                        )}
                      </div>
                    </div>
                  </div>
                </Card>
              )}
            </div>
          ) : (
            <Card className="p-12 text-center">
              <p className="text-gray-500">Select two models to compare</p>
            </Card>
          )}
        </div>
      </div>

      {/* Leaderboard */}
      <div className="px-8 py-8">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-bold text-gray-900">Models Leaderboard</h3>
            <div className="flex gap-3">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-40 bg-white border-gray-300">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent className="bg-white">
                  <SelectItem value="accuracy">Accuracy</SelectItem>
                  <SelectItem value="f1">F1 Score</SelectItem>
                  <SelectItem value="precision">Precision</SelectItem>
                  <SelectItem value="recall">Recall</SelectItem>
                  <SelectItem value="roc_auc">ROC-AUC</SelectItem>
                  <SelectItem value="r2">R² Score</SelectItem>
                  <SelectItem value="mse">MSE</SelectItem>
                  <SelectItem value="mae">MAE</SelectItem>
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
                className="border-gray-300"
              >
                {sortOrder === "asc" ? "↑ ASC" : "↓ DESC"}
              </Button>
            </div>
          </div>

          <div className="overflow-x-auto" key={filteredModels.toLocaleString()}>
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Model Name</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Type</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Accuracy</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">F1 Score</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">ROC-AUC</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">MSE</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">R² Score</th>
                </tr>
              </thead>
              <tbody>
                {filteredModels.map((model, idx) => (
                  <tr key={model.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gray-200 text-xs font-semibold mr-3">
                        {idx + 1}
                      </span>
                      {model.name}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">{model.model_type}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {model.metrics.accuracy !== undefined ? (model.metrics.accuracy * 100).toFixed(2) + "%" : "N/A"}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {model.metrics.f1 !== undefined ? model.metrics.f1.toFixed(4) : "N/A"}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {model.metrics.roc_auc !== undefined ? model.metrics.roc_auc.toFixed(4) : "N/A"}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {model.metrics.mse !== undefined ? model.metrics.mse.toFixed(4) : "N/A"}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {model.metrics.r2 !== undefined ? model.metrics.r2.toFixed(4) : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredModels.length === 0 && (
            <div className="text-center py-8">
              <p className="text-gray-500">No models available for comparison</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
