import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { Model } from "./models-content"
import { TrainingVisualizer } from "./training-visualizer"
import io from "socket.io-client"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api"
const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || "http://localhost:5000"

interface TrainModelDialogProps {
	modelIdInput?: number
	text?: string
	onTrainSuccess?: () => void
}

export function TrainModelDialog({ modelIdInput, text, onTrainSuccess }: TrainModelDialogProps) {
	const [open, setOpen] = useState(false)
	const [model, setModel] = useState({ id: -1, name: "Model not found", created_at: "", model_type: "linear_regression", dataset_id: -1, metrics: {} } as Model)
	const [error, setError] = useState("")
	const [isTraining, setIsTraining] = useState(false)
	const [showVisualizer, setShowVisualizer] = useState(false)
	const [earlyStopped, setEarlyStopped] = useState(false)
	const [startPaused, setStartPaused] = useState(false)

	// MLP hyperparameters with defaults
	const [hiddenLayers, setHiddenLayers] = useState("100")
	const [activation, setActivation] = useState("relu")
	const [solver, setSolver] = useState("adam")
	const [maxIter, setMaxIter] = useState("200")
	const [learningRate, setLearningRate] = useState("0.001")
	const [alpha, setAlpha] = useState("0.0001")

	const [regression, setRegression] = useState(false)
	const [progressPercent, setProgressPercent] = useState<number>(0)
	const [instantComplete, setInstantComplete] = useState<boolean>(false)
	const nonMlpSocketRef = useRef<ReturnType<typeof io> | null>(null)
	const hasShownAlertRef = useRef<boolean>(false)
	const simIntervalRef = useRef<number | null>(null)

	const getRegression = async (datasetId: number) => {
		try {
			const response = await fetch(`${API_URL}/datasets/${datasetId}`, {
				method: "GET",
				credentials: "include",
				headers: {
					"Accept": "application/json",
					"Content-Type": "application/json",
				},
			})
			if (!response.ok) {
				return
			}
			const data = await response.json()
			if (data.error) {
				return
			}
			setRegression(data.regression)
		} catch {
			console.log("Failed to fetch dataset for regression info")
		}
	}

	useEffect(() => {
		if (model.dataset_id !== -1) {
			getRegression(model.dataset_id)
		}
	}, [model.dataset_id])

	// hyperparameters state
	const [hyperparameters] = useState<Record<string, any>>({})

	const handleTrain = async () => {
		setError("")
		setIsTraining(true)
		setProgressPercent(0)
		setInstantComplete(false)
		hasShownAlertRef.current = false
		
		// For MLP, use WebSocket streaming
		if (model.model_type === 'mlp') {
			// First update model params with MLP configuration
			try {
				// Parse hidden layers (comma-separated numbers to tuple)
				const hiddenLayerSizes = hiddenLayers.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
				
				const mlpParams = {
					hidden_layer_sizes: hiddenLayerSizes,
					activation: activation,
					solver: solver,
					max_iter: parseInt(maxIter),
					learning_rate_init: parseFloat(learningRate),
					alpha: parseFloat(alpha)
				}
				
				// Update model params via PUT request
				const updateResponse = await fetch(`${API_URL}/models/${model.id}`, {
					method: "PUT",
					credentials: "include",
					headers: {
						"Accept": "application/json",
						"Content-Type": "application/json",
					},
					body: JSON.stringify({ params: mlpParams })
				})
				
				if (!updateResponse.ok) {
					throw new Error("Failed to update model parameters")
				}
				
				console.log("Updated MLP parameters:", mlpParams)
			} catch (err) {
				setError("Failed to configure MLP parameters")
				setIsTraining(false)
				return
			}
			
			setShowVisualizer(true)
			
			console.log("[TrainDialog] Creating WebSocket connection for MLP training")
			const socket = io(SOCKET_URL, {
				transports: ['websocket', 'polling']
			})
			
			socket.on('connect', () => {
				console.log('[TrainDialog] Socket connected, emitting start_training event')
				socket.emit('start_training', { model_id: model.id })
			})
			
			socket.on('training_complete', (data: any) => {
				console.log('[TrainDialog] Training complete:', data)
				setIsTraining(false)
				socket.disconnect()
				//onTrainSuccess?.()
			})
			
			socket.on('training_error', (data: any) => {
				console.error('[TrainDialog] Training error:', data)
				setError(data.message || 'Training failed')
				setIsTraining(false)
				socket.disconnect()
			})
			
			socket.on('connect_error', (err: any) => {
				console.error('[TrainDialog] Socket connection error:', err)
				setError('Failed to connect to training stream')
				setIsTraining(false)
			})
			
			return
		}
		
		// For other models, use regular POST and simulate progress locally until completion
		try {
			// start simulated progress (up to 95%)
			if (simIntervalRef.current !== null) {
				window.clearInterval(simIntervalRef.current)
				simIntervalRef.current = null
			}
			simIntervalRef.current = window.setInterval(() => {
				setProgressPercent((prev) => {
					// increment 1-3% per tick up to 95%
					const inc = Math.floor(Math.random() * 3) + 1
					return Math.min(95, prev + inc)
				})
			}, 250)

			const body: Record<string, any> = {}
			if (Object.keys(hyperparameters).length > 0) {
				body.hyperparams = hyperparameters
			}
			const response = await fetch(`${API_URL}/train/${model.id}`, {
				method: "POST",
				credentials: "include",
				headers: {
					"Accept": "application/json",
					"Content-Type": "application/json",
				},
				body: JSON.stringify(body),
			})
			if (!response.ok) {
				if (response.status === 400) {
					const data = await response.json()
					setError(data.error || "Invalid hyperparameters")
					setIsTraining(false)
					try { nonMlpSocketRef.current?.disconnect() } catch (_) {}
					nonMlpSocketRef.current = null
					return
				} else {
					setError(`Server error: ${response.status}`)
					setIsTraining(false)
					try { nonMlpSocketRef.current?.disconnect() } catch (_) {}
					nonMlpSocketRef.current = null
					return
				}
			}
			const data = await response.json()
			if (data.error) {
				setError(data.error)
				setIsTraining(false)
				if (simIntervalRef.current !== null) {
					window.clearInterval(simIntervalRef.current)
					simIntervalRef.current = null
				}
				return
			}
			// Complete: stop simulation, set to 100 (without animation), then show single alert after the UI paints 100%
			if (simIntervalRef.current !== null) {
				window.clearInterval(simIntervalRef.current)
				simIntervalRef.current = null
			}
			setInstantComplete(true)
			setProgressPercent(100)
			// Use double rAF to ensure the 100% width renders before showing alert
			requestAnimationFrame(() => {
				requestAnimationFrame(() => {
					if (!hasShownAlertRef.current) {
						hasShownAlertRef.current = true
						setIsTraining(false)
						try { alert('Training completed successfully!') } catch (_) {}
						setOpen(false)
						onTrainSuccess?.()
					}
				})
			})
		} catch (error) {
			setError("Failed to start training. Please try again.")
			setIsTraining(false)
			if (simIntervalRef.current !== null) {
				window.clearInterval(simIntervalRef.current)
				simIntervalRef.current = null
			}
			return
		}
	}

	const getModel = async () => {
		try {
			const response = await fetch(`${API_URL}/models/${modelIdInput}`, {
				method: "GET",
				credentials: "include",
				headers: {
					"Accept": "application/json",
					"Content-Type": "application/json",
				},
			})
			if (!response.ok) {
				return
			}
			const data = await response.json()
			if (data.error) {
				return
			}
			setModel(data)
			
			// Check if model was early stopped and whether it should resume paused
			if (data.early_stopped) {
				setEarlyStopped(true)
				// Set startPaused based on whether it was paused when early stopped
				// Default to true for backwards compatibility if was_paused is undefined
				const shouldStartPaused = data.was_paused !== undefined ? data.was_paused : false
				setStartPaused(shouldStartPaused)
				console.log(`Model was early stopped, will resume training ${shouldStartPaused ? 'paused' : 'running'}`)
			} else {
				setEarlyStopped(false)
				setStartPaused(false)
			}
			
			// Load existing params if available for MLP
			if (data.model_type === 'mlp' && data.params) {
				if (data.params.hidden_layer_sizes) {
					setHiddenLayers(Array.isArray(data.params.hidden_layer_sizes) 
						? data.params.hidden_layer_sizes.join(',') 
						: String(data.params.hidden_layer_sizes))
				}
				if (data.params.activation) setActivation(data.params.activation)
				if (data.params.solver) setSolver(data.params.solver)
				// Only load max_iter if NOT early stopped (user should set new epoch count)
				if (!data.early_stopped && data.params.max_iter) {
					setMaxIter(String(data.params.max_iter))
				}
				if (data.params.learning_rate_init) setLearningRate(String(data.params.learning_rate_init))
				if (data.params.alpha) setAlpha(String(data.params.alpha))
			}
		} catch {
			console.log("Failed to fetch model")
		}
	}

	useEffect(() => {
		if (open && modelIdInput) {
			getModel()
		}
	}, [open, modelIdInput])

	// Handle dialog close - reset state
	const handleOpenChange = (newOpen: boolean) => {
		setOpen(newOpen)
		
		// Reset state when closing dialog
		if (!newOpen) {
			setShowVisualizer(false)
			setIsTraining(false)
			setError("")
			setEarlyStopped(false)
			setStartPaused(false)
			setProgressPercent(0)
			hasShownAlertRef.current = false
			try { nonMlpSocketRef.current?.disconnect() } catch (_) {}
			nonMlpSocketRef.current = null
			if (simIntervalRef.current !== null) {
				window.clearInterval(simIntervalRef.current)
				simIntervalRef.current = null
			}
		}
	}

	return (
		<Dialog open={open} onOpenChange={handleOpenChange}>
			<DialogTrigger asChild>
				<Button className="bg-green-600 hover:bg-green-700 text-white">{text ? text : "Train Model"}</Button>
			</DialogTrigger>
			<DialogContent className={
				model.model_type === 'mlp' && showVisualizer 
					? "max-w-[98vw] w-[98vw] max-h-[92vh] overflow-y-auto"
					: model.model_type === 'mlp' 
						? "max-w-4xl"
						: ""
			}>
				<DialogHeader>
					<DialogTitle>Train {model.name}</DialogTitle>
					<DialogDescription>
						{model.model_type === 'mlp' 
							? showVisualizer 
								? "Neural network training with real-time progress visualization"
								: "Configure neural network hyperparameters before training"
							: "Enter the hyperparameters below to train the model."}
					</DialogDescription>
				</DialogHeader>
				
				{showVisualizer && model.model_type === 'mlp' ? (
					<div className="py-4">
						<TrainingVisualizer 
							modelId={model.id} 
							isVisible={showVisualizer} 
							regression={regression}
							startPaused={startPaused}
							onComplete={() => {
								setIsTraining(false)
								// Clear early stopped flag on successful completion
								setEarlyStopped(false)
							}}
						/>
					</div>
				) : (
					<div className="space-y-4 py-4">
						{model.model_type === "linear_regression" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for Linear Regression.</p>
						)}
						{model.model_type === "logistic_regression" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for Logistic Regression.</p>
						)}
						{model.model_type === "decision_tree" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for Decision Tree.</p>
						)}
						{model.model_type === "random_forest" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for Random Forest.</p>
						)}
						{model.model_type === "bagging" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for Bagging.</p>
						)}
						{model.model_type === "boosting" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for Boosting.</p>
						)}
						{model.model_type === "svm" && (
							<p className="text-sm text-gray-600">No hyperparameters to set for SVM.</p>
						)}
						{model.model_type === "mlp" && (
							<div className="space-y-4">
								{earlyStopped && (
									<div className="bg-blue-50 border border-blue-200 rounded-md p-3 mb-4">
										<p className="text-sm text-blue-800 font-medium">
											Resuming early-stopped training
										</p>
										<p className="text-xs text-blue-600 mt-1">
											Only the number of epochs can be changed. Other hyperparameters are locked to maintain model consistency.
										</p>
									</div>
								)}
								<div className="space-y-2">
									<Label htmlFor="hidden-layers">Hidden Layer Sizes</Label>
									<Input
										id="hidden-layers"
										type="text"
										value={hiddenLayers}
										onChange={(e) => setHiddenLayers(e.target.value)}
										placeholder="e.g., 100 or 100,50"
										disabled={earlyStopped}
										className={earlyStopped ? "bg-gray-100 cursor-not-allowed" : ""}
									/>
									<p className="text-xs text-gray-500">Comma-separated numbers for each hidden layer (e.g., "100,50" for two layers)</p>
								</div>
								
								<div className="space-y-2">
									<Label htmlFor="activation">Activation Function</Label>
									<Select value={activation} onValueChange={setActivation} disabled={earlyStopped}>
										<SelectTrigger id="activation" className={earlyStopped ? "bg-gray-100 cursor-not-allowed" : ""}>
											<SelectValue />
										</SelectTrigger>
										<SelectContent className="bg-white">
											<SelectItem value="relu" className="hover:bg-gray-200">ReLU</SelectItem>
											<SelectItem value="tanh" className="hover:bg-gray-200">Tanh</SelectItem>
											<SelectItem value="logistic" className="hover:bg-gray-200">Logistic (Sigmoid)</SelectItem>
											<SelectItem value="identity" className="hover:bg-gray-200">Identity (Linear)</SelectItem>
										</SelectContent>
									</Select>
								</div>
								
								<div className="space-y-2">
									<Label htmlFor="solver">Solver</Label>
									<Select value={solver} onValueChange={setSolver} disabled={earlyStopped}>
										<SelectTrigger id="solver" className={earlyStopped ? "bg-gray-100 cursor-not-allowed" : ""}>
											<SelectValue />
										</SelectTrigger>
										<SelectContent className="bg-white">
											<SelectItem value="adam" className="hover:bg-gray-200">Adam</SelectItem>
											<SelectItem value="sgd" className="hover:bg-gray-200">SGD</SelectItem>
											<SelectItem value="lbfgs" className="hover:bg-gray-200">L-BFGS</SelectItem>
										</SelectContent>
									</Select>
								</div>
								
								<div className="grid grid-cols-2 gap-4">
									<div className="space-y-2">
										<Label htmlFor="max-iter">Epochs</Label>
										<Input
											id="max-iter"
											type="number"
											value={maxIter}
											onChange={(e) => setMaxIter(e.target.value)}
											min="1"
										/>
									</div>
									
									<div className="space-y-2">
										<Label htmlFor="learning-rate">Learning Rate</Label>
										<Input
											id="learning-rate"
											type="number"
											step="0.0001"
											value={learningRate}
											onChange={(e) => setLearningRate(e.target.value)}
											min="0.0001"
											disabled={earlyStopped}
											className={earlyStopped ? "bg-gray-100 cursor-not-allowed" : ""}
										/>
									</div>
								</div>
								
								<div className="space-y-2">
									<Label htmlFor="alpha">L2 Regularization (Alpha)</Label>
									<Input
										id="alpha"
										type="number"
										step="0.0001"
										value={alpha}
										onChange={(e) => setAlpha(e.target.value)}
										min="0"
										disabled={earlyStopped}
										className={earlyStopped ? "bg-gray-100 cursor-not-allowed" : ""}
									/>
									<p className="text-xs text-gray-500">L2 penalty parameter (regularization strength)</p>
								</div>
							</div>
						)}
						{isTraining && model.model_type !== 'mlp' && (
							<div className="space-y-2">
								<Label>Training Progress</Label>
								<div className="w-full bg-gray-200 rounded h-3 overflow-hidden">
									<div 
										className="bg-blue-600 h-3"
										style={{ 
											width: `${progressPercent}%`, 
											transition: instantComplete ? 'none' : 'width 0.3s ease' 
										}} 
									/>
								</div>
								<div className="text-xs text-gray-700">{progressPercent}%</div>
							</div>
						)}
						{error && <p className="text-sm text-red-500">{error}</p>}
					</div>
				)}
				
				<div className="flex justify-end gap-3">
					{showVisualizer ? (
						!isTraining && (
							<Button 
								className="bg-green-600 hover:bg-green-700 text-white"
								onClick={() => {
									setOpen(false)
									setShowVisualizer(false)
									onTrainSuccess?.()
								}}
							>
								Done
							</Button>
						)
					) : (
						<>
							<Button variant="outline" className="bg-gray-100 hover:bg-gray-200" onClick={() => {
								setOpen(false)
								setShowVisualizer(false)
							}}>
								Cancel
							</Button>
							<Button
								className="bg-blue-600 hover:bg-blue-700 text-white"
								onClick={handleTrain}
								disabled={model.id === -1 || isTraining}
							>
								{isTraining ? "Training..." : "Train"}
							</Button>
						</>
					)}
				</div>
			</DialogContent>
		</Dialog>
	)
}
