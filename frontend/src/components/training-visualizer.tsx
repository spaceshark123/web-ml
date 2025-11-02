"use client"

import { useEffect, useRef, useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
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
  onCancel?: () => void
  onComplete?: () => void
}

export function TrainingVisualizer({ modelId, isVisible, regression, onCancel, onComplete }: TrainingVisualizerProps) {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([])
  const [isTraining, setIsTraining] = useState(true)
  const [isPaused, setIsPaused] = useState(false)
  const [isRegression, setIsRegression] = useState(regression)
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<any>(null)
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected" | "error">(
    "connecting",
  )
  // Zoom/Pan state shared across charts
  const [xDomain, setXDomain] = useState<[number, number] | null>(null)
  // Individual chart containers for accurate coordinate mapping
  const lossContainerRef = useRef<HTMLDivElement | null>(null)
  const metricContainerRef = useRef<HTMLDivElement | null>(null)
  const isPanningRef = useRef(false)
  const panStartRef = useRef<{ x: number; domain: [number, number]; width: number } | null>(null)

  const getDataBounds = () => {
    if (metrics.length === 0) return { min: 0, max: 1 }
    let min = metrics[0].epoch
    let max = metrics[0].epoch
    for (const m of metrics) {
      if (m.epoch < min) min = m.epoch
      if (m.epoch > max) max = m.epoch
    }
    return { min, max }
  }

  // Initialize domain when metrics first arrive or when domain becomes invalid
  useEffect(() => {
    if (metrics.length === 0) {
      setXDomain(null)
      return
    }
    if (!xDomain) {
      const { min, max } = getDataBounds()
      setXDomain([min, max])
    } else {
      // Ensure current domain is within data bounds
      const { min, max } = getDataBounds()
      const clampedMin = Math.max(xDomain[0], min)
      const clampedMax = Math.min(xDomain[1], max)
      if (clampedMax - clampedMin <= 0) {
        setXDomain([min, max])
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metrics])

  const clampDomain = (domain: [number, number]): [number, number] => {
    const { min, max } = getDataBounds()
    let [d0, d1] = domain
    // Normalize ordering
    if (d0 > d1) [d0, d1] = [d1, d0]
    // Clamp to data bounds
    d0 = Math.max(d0, min)
    d1 = Math.min(d1, max)
    // Enforce minimum visible span
    const minSpan = Math.max(1, Math.ceil((max - min) * 0.02)) // at least 1 epoch or 2% of range
    if (d1 - d0 < minSpan) {
      const mid = (d0 + d1) / 2
      d0 = Math.max(min, mid - minSpan / 2)
      d1 = Math.min(max, mid + minSpan / 2)
    }
    return [d0, d1]
  }

  // Create wheel handler bound to a specific container
  const makeWheelHandler = (container: React.RefObject<HTMLDivElement | null>): React.WheelEventHandler<HTMLDivElement> => (e) => {
    if (isTraining) return
    if (!xDomain || !container.current) return
    e.preventDefault()
    const rect = container.current.getBoundingClientRect()
    const width = rect.width
    const mouseX = e.clientX - rect.left
    const [d0, d1] = xDomain
    const span = d1 - d0
    const anchor = d0 + (mouseX / Math.max(1, width)) * span
    // Zoom factor: deltaY > 0 => zoom out, < 0 => zoom in
    const zoom = Math.exp(e.deltaY * 0.001)
    let newMin = anchor - (anchor - d0) * zoom
    let newMax = anchor + (d1 - anchor) * zoom
    const clamped = clampDomain([newMin, newMax])
    setXDomain(clamped)
  }

  const makeMouseDownHandler = (container: React.RefObject<HTMLDivElement | null>): React.MouseEventHandler<HTMLDivElement> => (e) => {
    if (isTraining) return
    if (!xDomain || !container.current) return
    isPanningRef.current = true
    const rect = container.current.getBoundingClientRect()
    panStartRef.current = { x: e.clientX, domain: xDomain, width: rect.width }
    // Change cursor
    container.current.style.cursor = 'grabbing'
  }

  const handleMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (isTraining) return
    if (!isPanningRef.current || !panStartRef.current) return
    const width = panStartRef.current.width || 1
    const dx = e.clientX - panStartRef.current.x
    const [s0, s1] = panStartRef.current.domain
    const span = s1 - s0
    const deltaEpoch = (dx / Math.max(1, width)) * span
    const newDomain: [number, number] = [s0 - deltaEpoch, s1 - deltaEpoch]
    setXDomain(clampDomain(newDomain))
  }

  const endPan = () => {
    // Reset cursors on both containers
    if (lossContainerRef.current) lossContainerRef.current.style.cursor = 'default'
    if (metricContainerRef.current) metricContainerRef.current.style.cursor = 'default'
    isPanningRef.current = false
    panStartRef.current = null
  }

  const handleMouseUp: React.MouseEventHandler<HTMLDivElement> = () => {
    endPan()
  }

  const handleMouseLeave: React.MouseEventHandler<HTMLDivElement> = () => {
    endPan()
  }

  const handleDoubleClick: React.MouseEventHandler<HTMLDivElement> = () => {
    const { min, max } = getDataBounds()
    setXDomain([min, max])
  }

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
      
      // Update regression flag from backend if provided
      if (data.regression !== undefined) {
        setIsRegression(data.regression)
      }
      
      setMetrics((prev) => {
        const updated = [...prev]
        const existingIdx = updated.findIndex((m) => m.epoch === data.epoch)
        
        // Normalize the metric field - backend may send 'metric', 'accuracy', or 'r2_score'
        const metricValue = data.metric !== undefined ? data.metric : (data.regression ? data.r2_score : data.accuracy)
        
        const newMetric = {
          epoch: data.epoch,
          loss: data.loss,
          metric: metricValue
        }
        
        if (existingIdx >= 0) {
          // Replace the existing entry completely
          updated[existingIdx] = newMetric
        } else {
          // Add new entry
          updated.push(newMetric)
        }
        return updated.sort((a, b) => a.epoch - b.epoch)
      })
    })

    socket.on('training_complete', (data: any) => {
      setIsTraining(false)
      const { min, max } = getDataBounds()
      setXDomain([min, max])
      setConnectionStatus("disconnected")
      console.log("[TrainingVisualizer] Training completed:", data.message)
      onComplete?.()
    })

    socket.on('training_paused', (data: any) => {
      setIsPaused(true)
      console.log("[TrainingVisualizer] Training paused:", data.model_id)
    })

    socket.on('training_resumed', (data: any) => {
      setIsPaused(false)
      console.log("[TrainingVisualizer] Training resumed:", data.model_id)
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
                {isTraining ? (isPaused ? "Training paused..." : "Training in progress...") : error ? "Training failed" : "Training completed"}
              </p>
              {hasData && (
                <p className="text-xs text-gray-600">
                  Epoch {lastMetric.epoch} • Loss: {lastMetric.loss.toFixed(4)}
                  {lastMetric.metric !== undefined && ` • ${isRegression ? "R²" : "Accuracy"}: ${(!isRegression ? (lastMetric.metric * 100).toFixed(2) + '%' : lastMetric.metric.toFixed(4))}`}
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
            {isTraining && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (socketRef.current) {
                      if (isPaused) {
                        socketRef.current.emit('resume_training', { model_id: modelId })
                      } else {
                        socketRef.current.emit('pause_training', { model_id: modelId })
                      }
                    }
                  }}
                  className="bg-yellow-50 hover:bg-yellow-100 text-yellow-700 border-yellow-200"
                >
                  {isPaused ? "Resume" : "Pause"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    try {
                      if (socketRef.current) {
                        socketRef.current.disconnect()
                      }
                    } finally {
                      // Reset all state
                      setIsTraining(false)
                      setIsPaused(false)
                      setMetrics([])
                      setError(null)
                      setConnectionStatus("disconnected")
                      // Call the parent's onCancel
                      onCancel?.()
                    }
                  }}
                  className="bg-red-50 hover:bg-red-100 text-red-600 border-red-200"
                >
                  Cancel
                </Button>
              </>
            )}
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 select-none">
          {/* Loss Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Training Loss</CardTitle>
              <CardDescription>Loss per epoch</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                ref={lossContainerRef}
                onWheel={makeWheelHandler(lossContainerRef)}
                onMouseDown={makeMouseDownHandler(lossContainerRef)}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
                onDoubleClick={handleDoubleClick}
                style={{ cursor: 'default' }}
                title={isTraining ? 'Zoom/pan enabled after training completes' : 'Scroll to zoom • Drag to pan • Double-click to reset'}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="epoch"
                      type="number"
                      allowDataOverflow
                      domain={xDomain ? [xDomain[0], xDomain[1]] : ['dataMin', 'dataMax']}
                      label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }}
                    />
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
              </div>
            </CardContent>
          </Card>

          {/* metric Chart */}
          {metrics[0].metric !== undefined && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training {isRegression ? "R² / Metric" : "Accuracy"}</CardTitle>
                <CardDescription>{isRegression ? "R² per epoch" : "Accuracy per epoch"}</CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  ref={metricContainerRef}
                  onWheel={makeWheelHandler(metricContainerRef)}
                  onMouseDown={makeMouseDownHandler(metricContainerRef)}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseLeave}
                  onDoubleClick={handleDoubleClick}
                  style={{ cursor: 'default' }}
                  title={isTraining ? 'Zoom/pan enabled after training completes' : 'Scroll to zoom • Drag to pan • Double-click to reset'}
                >
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="epoch"
                        type="number"
                        allowDataOverflow
                        domain={xDomain ? [xDomain[0], xDomain[1]] : ['dataMin', 'dataMax']}
                        label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }}
                      />
                      <YAxis label={{ value: isRegression ? "R²" : "Accuracy", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(value) => isRegression ? value.toString() : `${(Number(value) * 100).toFixed(2)}%`} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="metric"
                        stroke="#22c55e"
                        dot={false}
                        name={`Training ${isRegression ? "R²" : "Accuracy"}`}
                        strokeWidth={2}
                      />
                      {metrics[0].val_metric !== undefined && (
                        <Line
                          type="monotone"
                          dataKey="val_metric"
                          stroke="#16a34a"
                          dot={false}
                          name={`Validation ${isRegression ? "R²" : "Accuracy"}`}
                          strokeWidth={2}
                          strokeDasharray="5 5"
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
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
