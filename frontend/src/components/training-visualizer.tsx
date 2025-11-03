"use client"

import { useEffect, useRef, useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertCircle, Loader2, CheckCircle } from "lucide-react"
import io from "socket.io-client"
import DownloadableChart from "./downloadable-chart"

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
  onComplete?: () => void
  startPaused?: boolean
}

export function TrainingVisualizer({ modelId, isVisible, regression, onComplete, startPaused = false }: TrainingVisualizerProps) {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([])
  const [isTraining, setIsTraining] = useState(true)
  const [isPaused, setIsPaused] = useState(startPaused)
  const [isEarlyStopping, setIsEarlyStopping] = useState(false)
  const [isRegression, setIsRegression] = useState(regression)
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<any>(null)
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected" | "error">(
    "connecting",
  )
  // Zoom/Pan state shared across charts
  const [xDomain, setXDomain] = useState<[number, number] | null>(null)
  const [lossYDomain, setLossYDomain] = useState<[number, number] | null>(null)
  const [metricYDomain, setMetricYDomain] = useState<[number, number] | null>(null)
  // Individual chart containers for accurate coordinate mapping
  const lossContainerRef = useRef<HTMLDivElement | null>(null)
  const metricContainerRef = useRef<HTMLDivElement | null>(null)
  const isPanningRef = useRef(false)
  const panStartRef = useRef<{ x: number; domain: [number, number]; width: number } | null>(null)
  // Momentum tracking
  const velocityRef = useRef(0)
  const lastMoveTimeRef = useRef(0)
  const lastMoveXRef = useRef(0)
  const momentumAnimationRef = useRef<number | null>(null)
  // Keep current xDomain in a ref for momentum animation
  const xDomainRef = useRef<[number, number] | null>(null)

  // Sync xDomain state with ref
  useEffect(() => {
    xDomainRef.current = xDomain
  }, [xDomain])

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

  const getYDataBounds = (dataKey: 'loss' | 'metric', visibleMetrics: TrainingMetrics[]) => {
    if (visibleMetrics.length === 0) return { min: 0, max: 1 }
    let min = Infinity
    let max = -Infinity
    for (const m of visibleMetrics) {
      const value = dataKey === 'loss' ? m.loss : m.metric
      if (value !== undefined && Number.isFinite(value)) {
        if (value < min) min = value
        if (value > max) max = value
      }
    }
    if (min === Infinity || max === -Infinity) return { min: 0, max: 1 }
    // If all values are equal, create a small symmetric span around the value
    if (max === min) {
      const base = Math.abs(min)
      const eps = base > 0 ? base * 0.1 : 0.5 // at least ±0.5 around zero
      return { min: min - eps, max: max + eps }
    }
    // Add 10% padding and ensure a minimum span
    let padding = (max - min) * 0.1
    const minSpan = 1e-6
    if ((max - min + 2 * padding) < minSpan) {
      padding = (minSpan - (max - min)) / 2
    }
    return { min: min - padding, max: max + padding }
  }

  const getVisibleMetrics = () => {
    if (!xDomain || metrics.length === 0) return metrics
    return metrics.filter(m => m.epoch >= xDomain[0] && m.epoch <= xDomain[1])
  }

  // Add native wheel event listeners to prevent dialog scrolling AND handle zoom
  useEffect(() => {
    const lossContainer = lossContainerRef.current
    const metricContainer = metricContainerRef.current

    const handleWheel = (e: WheelEvent, container: HTMLDivElement) => {
      if (isTraining) return
      if (!xDomain || !container) return

      e.preventDefault()
      e.stopPropagation()

      const rect = container.getBoundingClientRect()
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

      // Update Y domains after zoom
      const visible = metrics.filter(m => m.epoch >= clamped[0] && m.epoch <= clamped[1])
      const lossYBounds = getYDataBounds('loss', visible)
      setLossYDomain([lossYBounds.min, lossYBounds.max])
      if (visible.length > 0 && visible[0].metric !== undefined) {
        const metricYBounds = getYDataBounds('metric', visible)
        setMetricYDomain([metricYBounds.min, metricYBounds.max])
      }
    }

    const lossWheelHandler = (e: WheelEvent) => lossContainer && handleWheel(e, lossContainer)
    const metricWheelHandler = (e: WheelEvent) => metricContainer && handleWheel(e, metricContainer)

    if (lossContainer) {
      lossContainer.addEventListener('wheel', lossWheelHandler, { passive: false })
    }
    if (metricContainer) {
      metricContainer.addEventListener('wheel', metricWheelHandler, { passive: false })
    }

    return () => {
      if (lossContainer) {
        lossContainer.removeEventListener('wheel', lossWheelHandler)
      }
      if (metricContainer) {
        metricContainer.removeEventListener('wheel', metricWheelHandler)
      }
    }
  }, [isTraining, xDomain, metrics])

  // Initialize and auto-extend domain while training so X-axis labels advance each epoch
  useEffect(() => {
    if (metrics.length === 0) {
      // only update when needed
      if (xDomain !== null) setXDomain(null)
      if (lossYDomain !== null) setLossYDomain(null)
      if (metricYDomain !== null) setMetricYDomain(null)
      return
    }

    const { min, max } = getDataBounds()
    const newDomain: [number, number] = [min, max]
    const prev = xDomainRef.current

    // Only update xDomain when numeric bounds actually change to avoid spurious state updates
    if (isTraining) {
      if (!prev || prev[0] !== newDomain[0] || prev[1] !== newDomain[1]) {
        setXDomain(newDomain)
      }
    } else {
      if (!prev) {
        setXDomain(newDomain)
      } else {
        const clampedMin = Math.max(prev[0], min)
        const clampedMax = Math.min(prev[1], max)
        if (clampedMax - clampedMin <= 0 || clampedMin !== prev[0] || clampedMax !== prev[1]) {
          setXDomain(newDomain)
        }
      }
    }

    // Update Y domains based on visible data
    const visible = getVisibleMetrics()
    const lossYBounds = getYDataBounds('loss', visible)
    setLossYDomain([lossYBounds.min, lossYBounds.max])
    if (visible.length > 0 && visible[0].metric !== undefined) {
      const metricYBounds = getYDataBounds('metric', visible)
      setMetricYDomain([metricYBounds.min, metricYBounds.max])
    }
    // dependencies: only metrics and isTraining (xDomain compared via ref)
  }, [metrics, isTraining])

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

  const makeMouseDownHandler = (container: React.RefObject<HTMLDivElement | null>): React.MouseEventHandler<HTMLDivElement> => (e) => {
    if (isTraining) return
    if (!container.current) return

    // Cancel any ongoing momentum animation
    if (momentumAnimationRef.current !== null) {
      cancelAnimationFrame(momentumAnimationRef.current)
      momentumAnimationRef.current = null
    }

    // Use the ref to get the most current domain value (in case momentum just updated it)
    const currentDomain = xDomainRef.current
    if (!currentDomain) return

    isPanningRef.current = true
    const rect = container.current.getBoundingClientRect()
    panStartRef.current = { x: e.clientX, domain: currentDomain, width: rect.width }
    velocityRef.current = 0
    lastMoveTimeRef.current = Date.now()
    lastMoveXRef.current = e.clientX
    // Change cursor
    container.current.style.cursor = 'grabbing'
  }

  const handleMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (isTraining) return
    if (!isPanningRef.current || !panStartRef.current) return

    const now = Date.now()
    const dt = now - lastMoveTimeRef.current

    if (dt > 0) {
      const dx = e.clientX - lastMoveXRef.current
      velocityRef.current = dx / dt // pixels per millisecond
    }

    lastMoveTimeRef.current = now
    lastMoveXRef.current = e.clientX

    const width = panStartRef.current.width || 1
    const dx = e.clientX - panStartRef.current.x
    const [s0, s1] = panStartRef.current.domain
    const span = s1 - s0
    const deltaEpoch = (dx / Math.max(1, width)) * span
    const newDomain: [number, number] = [s0 - deltaEpoch, s1 - deltaEpoch]
    const clamped = clampDomain(newDomain)
    setXDomain(clamped)

    // Update Y domains during pan
    const visible = metrics.filter(m => m.epoch >= clamped[0] && m.epoch <= clamped[1])
    const lossYBounds = getYDataBounds('loss', visible)
    setLossYDomain([lossYBounds.min, lossYBounds.max])
    if (visible.length > 0 && visible[0].metric !== undefined) {
      const metricYBounds = getYDataBounds('metric', visible)
      setMetricYDomain([metricYBounds.min, metricYBounds.max])
    }
  }

  const applyMomentum = () => {
    const currentDomain = xDomainRef.current
    // If momentum is effectively done or domain not ready, finalize and recalc Y once
    if (!currentDomain || Math.abs(velocityRef.current) < 0.01) {
      if (currentDomain) {
        const visibleEnd = metrics.filter(m => m.epoch >= currentDomain[0] && m.epoch <= currentDomain[1])
        const lossYBoundsEnd = getYDataBounds('loss', visibleEnd)
        setLossYDomain([lossYBoundsEnd.min, lossYBoundsEnd.max])
        if (visibleEnd.length > 0 && visibleEnd[0].metric !== undefined) {
          const metricYBoundsEnd = getYDataBounds('metric', visibleEnd)
          setMetricYDomain([metricYBoundsEnd.min, metricYBoundsEnd.max])
        }
      }
      momentumAnimationRef.current = null
      return
    }

    const { min, max } = getDataBounds()
    const [d0, d1] = currentDomain
    const span = d1 - d0

    // Apply velocity (convert from pixels/ms to epochs, assuming same width as last pan)
    const width = panStartRef.current?.width || 1
    // Reduce momentum scale for a gentler glide
    const deltaEpoch = -(velocityRef.current * 0.5) * (span / width) // was 16

    let newMin = d0 + deltaEpoch
    let newMax = d1 + deltaEpoch

    // Clamp to bounds
    if (newMin < min) {
      newMin = min
      newMax = min + span
      velocityRef.current = 0
    }
    if (newMax > max) {
      newMax = max
      newMin = max - span
      velocityRef.current = 0
    }

    const clamped = clampDomain([newMin, newMax])
    setXDomain(clamped)

    // Update Y domains during momentum
    const visible = metrics.filter(m => m.epoch >= clamped[0] && m.epoch <= clamped[1])
    const lossYBounds = getYDataBounds('loss', visible)
    setLossYDomain([lossYBounds.min, lossYBounds.max])
    if (visible.length > 0 && visible[0].metric !== undefined) {
      const metricYBounds = getYDataBounds('metric', visible)
      setMetricYDomain([metricYBounds.min, metricYBounds.max])
    }

    // Apply stronger friction to shorten momentum
    velocityRef.current *= 0.5 // was 0.95

    // Continue animation
    momentumAnimationRef.current = requestAnimationFrame(applyMomentum)
  }

  const endPan = () => {
    // Reset cursors on both containers
    if (lossContainerRef.current) lossContainerRef.current.style.cursor = 'default'
    if (metricContainerRef.current) metricContainerRef.current.style.cursor = 'default'

    // Recalculate Y domains at the end of the drag based on the final domain
    const currentDomain = xDomainRef.current
    if (currentDomain) {
      const visibleEnd = metrics.filter(m => m.epoch >= currentDomain[0] && m.epoch <= currentDomain[1])
      const lossYBoundsEnd = getYDataBounds('loss', visibleEnd)
      setLossYDomain([lossYBoundsEnd.min, lossYBoundsEnd.max])
      if (visibleEnd.length > 0 && visibleEnd[0].metric !== undefined) {
        const metricYBoundsEnd = getYDataBounds('metric', visibleEnd)
        setMetricYDomain([metricYBoundsEnd.min, metricYBoundsEnd.max])
      }
    }

    if (isPanningRef.current && Math.abs(velocityRef.current) > 0.1) {
      // Start momentum animation
      momentumAnimationRef.current = requestAnimationFrame(applyMomentum)
    }

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
    // Cancel any momentum animation
    if (momentumAnimationRef.current !== null) {
      cancelAnimationFrame(momentumAnimationRef.current)
      momentumAnimationRef.current = null
    }
    velocityRef.current = 0

    const { min, max } = getDataBounds()
    setXDomain([min, max])

    // Reset Y domains to full data range
    const lossYBounds = getYDataBounds('loss', metrics)
    setLossYDomain([lossYBounds.min, lossYBounds.max])
    if (metrics.length > 0 && metrics[0].metric !== undefined) {
      const metricYBounds = getYDataBounds('metric', metrics)
      setMetricYDomain([metricYBounds.min, metricYBounds.max])
    }
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

      // No need to manually pause here - backend will handle paused state for early-stopped models
      // and will send training_paused event if needed
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

    socket.on('training_early_stopped', (data: any) => {
      setIsEarlyStopping(true)
      console.log("[TrainingVisualizer] Early stop acknowledged:", data.model_id)
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
      if (momentumAnimationRef.current !== null) {
        cancelAnimationFrame(momentumAnimationRef.current)
      }
      setIsEarlyStopping(false)
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
                {isTraining ? (isPaused ? "Training paused..." : isEarlyStopping ? "Stopping after current epoch..." : "Training in progress...") : error ? "Training failed" : "Training completed"}
              </p>
              {hasData && (
                <p className="text-xs text-gray-600">
                  Epoch {lastMetric.epoch} • Loss: {lastMetric.loss.toFixed(4)}
                  {lastMetric.metric !== undefined && ` • ${isRegression ? "MSE" : "Accuracy"}: ${(!isRegression ? (lastMetric.metric * 100).toFixed(2) + '%' : lastMetric.metric.toFixed(4))}`}
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
                  disabled={isEarlyStopping}
                >
                  {isPaused ? "Resume" : "Pause"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (socketRef.current && !isEarlyStopping) {
                      setIsEarlyStopping(true)
                      socketRef.current.emit('early_stop_training', { model_id: modelId })
                    }
                  }}
                  className="bg-orange-50 hover:bg-orange-100 text-orange-600 border-orange-200"
                  disabled={isEarlyStopping}
                >
                  {isEarlyStopping ? "Stopping..." : "Early Stop"}
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
        <div className="grid grid-cols-1 gap-4 select-none">
          {/* Loss Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Training Loss</CardTitle>
              <CardDescription>Loss per epoch</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                ref={lossContainerRef}
                onMouseDown={makeMouseDownHandler(lossContainerRef)}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
                onDoubleClick={handleDoubleClick}
                style={{ cursor: 'default' }}
                title={isTraining ? 'Zoom/pan enabled after training completes' : 'Scroll to zoom • Drag to pan • Double-click to reset'}
              >
                <DownloadableChart>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="epoch"
                        type="number"
                        allowDataOverflow
                        domain={xDomain ? [xDomain[0], xDomain[1]] : ['dataMin', 'dataMax']}
                        label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }}
                      />
                      <YAxis
                        label={{ value: "Loss", angle: -90, position: "left" }}
                        domain={lossYDomain ? [lossYDomain[0], lossYDomain[1]] : ['auto', 'auto']}
                        tickFormatter={(v) => Number(v).toFixed(4)}
                      />
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
                </DownloadableChart>
              </div>
            </CardContent>
          </Card>

          {/* metric Chart */}
          {metrics[0].metric !== undefined && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training {isRegression ? "MSE" : "Accuracy"}</CardTitle>
                <CardDescription>{isRegression ? "MSE per epoch" : "Accuracy per epoch"}</CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  ref={metricContainerRef}
                  onMouseDown={makeMouseDownHandler(metricContainerRef)}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseLeave}
                  onDoubleClick={handleDoubleClick}
                  style={{ cursor: 'default' }}
                  title={isTraining ? 'Zoom/pan enabled after training completes' : 'Scroll to zoom • Drag to pan • Double-click to reset'}
                >
                  <DownloadableChart>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={metrics} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="epoch"
                          type="number"
                          allowDataOverflow
                          domain={xDomain ? [xDomain[0], xDomain[1]] : ['dataMin', 'dataMax']}
                          label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }}
                        />
                        <YAxis
                          label={{ value: isRegression ? "MSE" : "Accuracy", angle: -90, position: "left" }}
                          domain={metricYDomain ? [metricYDomain[0], metricYDomain[1]] : ['auto', 'auto']}
                          tickFormatter={(v) => isRegression ? Number(v).toFixed(4) : `${(Number(v) * 100).toFixed(2)}%`}
                        />
                        <Tooltip formatter={(value) => isRegression ? value.toString() : `${(Number(value) * 100).toFixed(2)}%`} />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="metric"
                          stroke="#22c55e"
                          dot={false}
                          name={`Training ${isRegression ? "MSE" : "Accuracy"}`}
                          strokeWidth={2}
                        />
                        {metrics[0].val_metric !== undefined && (
                          <Line
                            type="monotone"
                            dataKey="val_metric"
                            stroke="#16a34a"
                            dot={false}
                            name={`Validation ${isRegression ? "MSE" : "Accuracy"}`}
                            strokeWidth={2}
                            strokeDasharray="5 5"
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </DownloadableChart>
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
