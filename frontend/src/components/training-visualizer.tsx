"use client"

import { useEffect, useRef, useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, Loader2, CheckCircle } from "lucide-react"
import io from "socket.io-client"

export interface TrainingMetrics {
  epoch: number
  loss: number
  metric?: number
  val_loss?: number
  val_metric?: number
}

interface TrainingVisualizerProps {
  regression: boolean
  modelId: number
  isVisible: boolean
}

export function TrainingVisualizer({ modelId, isVisible, regression }: TrainingVisualizerProps) {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([])
  const [isTraining, setIsTraining] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<any>(null)
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected" | "error">(
    "connecting",
  )

  useEffect(() => {
    if (!isVisible || !modelId) return

    console.log(`[TrainingVisualizer] Initializing for model ${modelId}`)

    setMetrics([])
    setError(null)
    setIsTraining(true)
    setConnectionStatus("connecting")

    const socket = io("http://localhost:5000", {
      transports: ['websocket', 'polling']
    })

    socket.on('connect', () => {
      setConnectionStatus("connected")
      console.log("[TrainingVisualizer] Socket connected successfully")
    })

    socket.on('training_metrics', (data: any) => {
      console.log("[TrainingVisualizer] Received training_metrics:", data)
      setMetrics((prev) => {
        const updated = [...prev]
        const existingIdx = updated.findIndex((m) => m.epoch === data.epoch)
        if (existingIdx >= 0) {
          updated[existingIdx] = { ...updated[existingIdx], ...data }
        } else {
          updated.push({
            epoch: data.epoch,
            loss: data.loss,
            metric: data.metric
          })
        }
        return updated.sort((a, b) => a.epoch - b.epoch)
      })
    })

    socket.on('training_complete', (data: any) => {
      setIsTraining(false)
      setConnectionStatus("disconnected")
      console.log("[TrainingVisualizer] Training completed:", data.message)
    })

    socket.on('training_error', (data: any) => {
      setError(data.message)
      setIsTraining(false)
      setConnectionStatus("error")
      console.error("[TrainingVisualizer] Training error:", data.message)
    })

    socket.on('connect_error', (err: any) => {
      setConnectionStatus("error")
      setError("Failed to connect to training stream")
      console.error("[TrainingVisualizer] Socket connection error:", err)
    })

    socket.on('disconnect', () => {
      console.log("[TrainingVisualizer] Socket disconnected")
      if (connectionStatus !== "error") {
        setConnectionStatus("disconnected")
      }
    })

    socketRef.current = socket

    return () => {
      console.log("[TrainingVisualizer] Cleaning up socket connection")
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
    }
  }, [modelId, isVisible])

  if (!isVisible) return null

  const hasData = metrics.length > 0
  const lastMetric = metrics[metrics.length - 1]

  return (
    <div className="space-y-4">
      {/* Status Bar */}
      <Card className="border-blue-200 bg-blue-50">
        <CardContent className="pt-6">
          <div className="flex items-center gap-3">
            {isTraining ? (
              <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
            ) : error ? (
              <AlertCircle className="w-5 h-5 text-red-600" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green-600" />
            )}
            <div className="flex-1">
              <p className="font-medium text-sm">
                {isTraining ? "Training in progress..." : error ? "Training failed" : "Training completed"}
              </p>
              {hasData && (
                <p className="text-xs text-gray-600">
                  Epoch {lastMetric.epoch} • Loss: {lastMetric.loss.toFixed(4)}
                  {lastMetric.metric !== undefined && ` • ${regression ? "MSE" : "Accuracy"}: ${(!regression ? (lastMetric.metric * 100).toFixed(2) + '%' : lastMetric.metric.toFixed(4))}`}
                </p>
              )}
            </div>
            <div
              className={`px-3 py-1 rounded text-xs font-medium ${connectionStatus === "connected"
                  ? "bg-green-100 text-green-800"
                  : connectionStatus === "connecting"
                    ? "bg-yellow-100 text-yellow-800"
                    : "bg-red-100 text-red-800"
                }`}
            >
              {connectionStatus === "connected"
                ? "Connected"
                : connectionStatus === "connecting"
                  ? "Connecting..."
                  : "Disconnected"}
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <p className="text-sm text-red-800">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Metrics Charts */}
      {hasData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Loss Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Training Loss</CardTitle>
              <CardDescription>Loss per epoch</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }} />
                  <YAxis label={{ value: "Loss", angle: -90, position: "insideLeft" }} />
                  <Tooltip formatter={(value) => value.toString()} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#ef4444"
                    dot={false}
                    name="Training Loss"
                    strokeWidth={2}
                  />
                  {metrics[0].val_loss !== undefined && (
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      stroke="#f97316"
                      dot={false}
                      name="Validation Loss"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* metric Chart */}
          {metrics[0].metric !== undefined && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training { regression ? "MSE" : "Accuracy" }</CardTitle>
                <CardDescription>{ regression ? "MSE per epoch" : "Accuracy per epoch" }</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }} />
                    <YAxis label={{ value: regression ? "MSE" : "Accuracy", angle: -90, position: "insideLeft" }} />
                    <Tooltip formatter={(value) => `${(value.toString())}`} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="metric"
                      stroke="#22c55e"
                      dot={false}
                      name={`Training ${regression ? "MSE" : "Accuracy"}`}
                      strokeWidth={2}
                    />
                    {metrics[0].val_metric !== undefined && (
                      <Line
                        type="monotone"
                        dataKey="val_metric"
                        stroke="#16a34a"
                        dot={false}
                        name={`Validation ${regression ? "MSE" : "Accuracy"}`}
                        strokeWidth={2}
                        strokeDasharray="5 5"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!hasData && isTraining && (
        <Card className="border-dashed">
          <CardContent className="pt-12 pb-12 flex flex-col items-center justify-center">
            <Loader2 className="w-8 h-8 text-gray-400 animate-spin mb-3" />
            <p className="text-gray-500">Waiting for training metrics...</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
