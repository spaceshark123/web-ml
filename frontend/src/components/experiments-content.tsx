"use client"

import { useState, useEffect } from "react"
import { ArrowLeft, Trash2 } from "lucide-react"
import { Button } from "./ui/button"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "./ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Link, useSearchParams } from "react-router-dom"
import { API_BASE_URL } from "@/constants"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar } from "recharts"
import { Label } from "@/components/ui/label"
import type { Dataset } from "./datasets-content"
import type { Model } from "./models-content"

type CurvePoint = { fpr?: number; tpr?: number; precision?: number; recall?: number }
interface ExperimentsResponse {
	model_id: number
	model_name: string
	model_type: string
	type: 'classification' | 'regression'
	metrics: Record<string, any> & {
		accuracy?: number; precision?: number; recall?: number; f1?: number; roc_auc?: number; pr_auc?: number;
		mse?: number; mae?: number; r2?: number;
		roc_curve?: CurvePoint[]
		pr_curve?: CurvePoint[]
	}
	confusion_matrix?: { labels: string[]; matrix: number[][] }
	imbalance?: { minority_class_percentage: number; imbalance_ratio: number; is_imbalanced: boolean }
	shap?: { feature_importance: { feature: string; importance: number }[]; total_features?: number; error?: string }
}

interface SavedExperiment {
	id: string
	timestamp: number
	data: ExperimentsResponse
}

// LocalStorage helpers for experiment history
const EXPERIMENT_HISTORY_KEY = 'web-ml-experiment-history'

function saveExperimentToHistory(data: ExperimentsResponse) {
	const history = loadExperimentHistory()
	const newEntry: SavedExperiment = {
		id: `${data.model_id}-${Date.now()}`,
		timestamp: Date.now(),
		data
	}
	// Add to front, keep all experiments (unlimited storage)
	const updated = [newEntry, ...history]
	localStorage.setItem(EXPERIMENT_HISTORY_KEY, JSON.stringify(updated))
}

function loadExperimentHistory(): SavedExperiment[] {
	try {
		const raw = localStorage.getItem(EXPERIMENT_HISTORY_KEY)
		return raw ? JSON.parse(raw) : []
	} catch {
		return []
	}
}

function deleteExperimentFromHistory(id: string): SavedExperiment[] {
	try {
		const raw = localStorage.getItem(EXPERIMENT_HISTORY_KEY)
		const history: SavedExperiment[] = raw ? JSON.parse(raw) : []
		const updated = history.filter((h) => h.id !== id)
		localStorage.setItem(EXPERIMENT_HISTORY_KEY, JSON.stringify(updated))
		return updated
	} catch {
		return []
	}
}

// Reserved for future cross-page usage:
// function deleteExperimentsForModel(modelId: number): SavedExperiment[] {
//     try {
//         const raw = localStorage.getItem(EXPERIMENT_HISTORY_KEY)
//         const history: SavedExperiment[] = raw ? JSON.parse(raw) : []
//         const updated = history.filter((h) => h.data?.model_id !== modelId)
//         localStorage.setItem(EXPERIMENT_HISTORY_KEY, JSON.stringify(updated))
//         return updated
//     } catch {
//         return []
//     }
// }

// Custom tooltip for ROC curves was inlined at usage sites to avoid scope/runtime issues.

// Custom tooltip for PR curves that shows all series at the cursor's Recall
function PrTooltip({ active, payload, label }: any) {
	if (!active || !payload || payload.length === 0) return null
	const dataPoints = payload
	const cursorRecall = typeof label === 'number' ? label : (dataPoints[0]?.payload?.recall ?? 0)
	return (
		<div className="rounded-md bg-white/95 border border-gray-200 shadow p-2 text-xs">
			<div className="font-medium text-gray-700 mb-1">Recall: {Number(cursorRecall).toFixed(3)}</div>
			{dataPoints.map((point: any, idx: number) => (
				<div key={idx} className="flex items-center gap-2">
					<span className="inline-block w-2 h-2 rounded-sm" style={{ background: point.color }} />
					<span className="text-gray-600">{point.name}:</span>
					<span className="text-gray-900">Precision {Number(point.value).toFixed(3)}</span>
				</div>
			))}
		</div>
	)
}

// Helper function to merge multiclass ROC curves into a single dataset
// This enables synchronized tooltips across all curves at the same FPR
function mergeRocCurves(rocCurvesOvr: Array<{class_label: string; curve: Array<{fpr: number; tpr: number}>}>) {
	// Collect all unique FPR values from all curves
	const fprSet = new Set<number>()
	rocCurvesOvr.forEach(cls => {
		cls.curve.forEach(point => fprSet.add(point.fpr))
	})
	const sortedFprs = Array.from(fprSet).sort((a, b) => a - b)
	
	// For each unique FPR, interpolate TPR for each class
	return sortedFprs.map(fpr => {
		const point: any = { fpr }
		rocCurvesOvr.forEach(cls => {
			// Find or interpolate the TPR at this FPR
			const curve = cls.curve
			let tpr = 0
			
			// Find the two points that bracket this FPR
			for (let i = 0; i < curve.length; i++) {
				if (curve[i].fpr === fpr) {
					tpr = curve[i].tpr
					break
				} else if (i > 0 && curve[i-1].fpr < fpr && curve[i].fpr > fpr) {
					// Linear interpolation between two points
					const t = (fpr - curve[i-1].fpr) / (curve[i].fpr - curve[i-1].fpr)
					tpr = curve[i-1].tpr + t * (curve[i].tpr - curve[i-1].tpr)
					break
				} else if (i === curve.length - 1) {
					// Use last point if beyond range
					tpr = curve[i].tpr
				}
			}
			
			point[cls.class_label] = tpr
		})
		return point
	})
}

// Helper to merge multiclass PR curves into a single dataset keyed by recall
function mergePrCurves(prCurvesOvr: Array<{class_label: string; curve: Array<{recall: number; precision: number}>}>) {
	const recallSet = new Set<number>()
	prCurvesOvr.forEach(cls => {
		cls.curve.forEach(pt => recallSet.add(pt.recall))
	})
	const sortedRecalls = Array.from(recallSet).sort((a, b) => a - b)

	return sortedRecalls.map(recall => {
		const point: any = { recall }
		prCurvesOvr.forEach(cls => {
			const curve = cls.curve
			let prec = 0
			for (let i = 0; i < curve.length; i++) {
				if (curve[i].recall === recall) {
					prec = curve[i].precision
					break
				} else if (i > 0 && curve[i-1].recall < recall && curve[i].recall > recall) {
					const t = (recall - curve[i-1].recall) / (curve[i].recall - curve[i-1].recall)
					prec = curve[i-1].precision + t * (curve[i].precision - curve[i-1].precision)
					break
				} else if (i === curve.length - 1) {
					prec = curve[i].precision
				}
			}
			point[cls.class_label] = prec
		})
		return point
	})
}


export function ExperimentsContent() {
	const [models, setModels] = useState<Model[]>([])
	const [datasets, setDatasets] = useState<Dataset[]>([])
	const [type, setType] = useState<'model' | 'dataset'>('model')
	const [selectedDataset, setSelectedDataset] = useState<string>("")
	const [selectedModel, setSelectedModel] = useState<string>("")
	const [experimentData, setExperimentData] = useState<ExperimentsResponse | null>(null)
	const [loading, setLoading] = useState(false)
	const [shapProgress, setShapProgress] = useState<number>(0)
	const [shapPollingId, setShapPollingId] = useState<number | null>(null)
	const [experimentHistory, setExperimentHistory] = useState<SavedExperiment[]>([])
	const [searchQuery, setSearchQuery] = useState<string>("")
	const [showAllHistory, setShowAllHistory] = useState<boolean>(false)
	const [searchParams] = useSearchParams()

	useEffect(() => {
		fetchModels()
		fetchDatasets()
		// Load experiment history from localStorage on mount
		const history = loadExperimentHistory()
		setExperimentHistory(history)
		
		// Check if experiment ID is in URL query params (from dashboard link)
		const experimentId = searchParams.get('id')
		if (experimentId) {
			const experiment = history.find(exp => exp.id === experimentId)
			if (experiment) {
				setExperimentData(experiment.data)
			}
		}

		// If a model_id is provided, preselect it in the Model selector
		const modelId = searchParams.get('model_id')
		if (modelId) {
			setType('model')
			setSelectedModel(modelId)
		}
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

	const visualizeDataset = async () => {
		setExperimentData(null)
		if (!selectedDataset) return
		setLoading(true)
		// Placeholder: dataset visualizations not yet implemented
		setTimeout(() => setLoading(false), 800)
	}

	const evaluateModel = async () => {
		setExperimentData(null)
		if (!selectedModel) return
		
		// Clear any existing polling interval
		if (shapPollingId !== null) {
			window.clearInterval(shapPollingId)
			setShapPollingId(null)
		}
		setShapProgress(0)
		
		setLoading(true)
		
		// Capture the model ID we're evaluating to prevent stale closures
		const modelForPolling = selectedModel
		
		try {
			const res = await fetch(`${API_BASE_URL}/models/${selectedModel}/experiments?shap_async=1`, {
				credentials: 'include',
				headers: { 'Content-Type': 'application/json' },
			})
			if (!res.ok) {
				// Handle untrained/missing model gracefully
				try {
					const err = await res.json()
					if (err && err.error) {
						// Use the specified alert message
						window.alert("Error: Model doesn't exist or has not trained.")
						setLoading(false)
						return
					}
				} catch {}
				window.alert("Error: Model doesn't exist or has not trained")
				setLoading(false)
				return
			}
			const data: ExperimentsResponse = await res.json()
			console.log('Experiment data received:', data)
			console.log('ROC curve data:', data.metrics?.roc_curve)
			console.log('ROC curves OvR:', (data.metrics as any)?.roc_curves_ovr)
			setExperimentData(data)
			
			// Save experiment to history
			saveExperimentToHistory(data)
			setExperimentHistory(loadExperimentHistory())
			
			// If SHAP is pending, start polling progress
			if (data && (data as any).shap && (data as any).shap.status === 'pending') {
				setShapProgress(0)
				
				// Begin polling every 1 second for smoother progress updates
				const intervalId = window.setInterval(async () => {
					// Stop polling if user switched to a different model
					if (selectedModel !== modelForPolling) {
						window.clearInterval(intervalId)
						setShapPollingId(null)
						return
					}
					
					try {
						const r = await fetch(`${API_BASE_URL}/models/${modelForPolling}/experiments/shap`, {
							credentials: 'include',
							headers: { 'Content-Type': 'application/json' },
						})
						const stat = await r.json()
						console.log('SHAP polling response:', stat)
						
						if (typeof stat.progress === 'number') setShapProgress(stat.progress)
						
						if (stat.status === 'done' && stat.shap) {
							console.log('SHAP computation complete, updating data:', stat.shap)
							setExperimentData((prev) => {
								const updated = prev ? { ...prev, shap: stat.shap } : prev
								console.log('Updated experiment data:', updated)
								return updated
							})
							setShapProgress(0)
							window.clearInterval(intervalId)
							setShapPollingId(null)
						} else if (stat.status === 'error') {
							setExperimentData((prev) => prev ? { ...prev, shap: { feature_importance: [], error: stat.error } } : prev)
							setShapProgress(0)
							window.clearInterval(intervalId)
							setShapPollingId(null)
						} else if (stat.status === 'none') {
							// Job doesn't exist, stop polling
							window.clearInterval(intervalId)
							setShapPollingId(null)
							setShapProgress(0)
							setExperimentData((prev) => prev ? { ...prev, shap: { feature_importance: [], error: 'SHAP job not found' } } : prev)
						}
					} catch (e) {
						console.error('SHAP polling error:', e)
						// Stop polling on error
						window.clearInterval(intervalId)
						setShapPollingId(null)
					}
				}, 1000) // Poll every 1 second for smoother progress updates				setShapPollingId(intervalId)
			} else {
				setShapProgress(0)
			}
		} catch (e) {
			console.error('Failed to evaluate model', e)
		} finally {
			setLoading(false)
		}
	}
	
	// Cleanup polling on unmount
	useEffect(() => {
		return () => {
			if (shapPollingId !== null) {
				window.clearInterval(shapPollingId)
			}
		}
	}, [shapPollingId])

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
				<h1 className="text-4xl font-bold text-gray-900 mb-2">Experiments</h1>
				<p className="text-gray-500">Evaluate models, visualize datasets, and observe metrics</p>
			</div>

			<div className="p-8 grid grid-cols-1 lg:grid-cols-4 gap-8">
				{/* Selection */}
				<div className="lg:col-span-1 space-y-4">
					<Card className="p-6">
						<h3 className="text-lg font-semibold mb-4">Select Model/Dataset</h3>

						<div className="space-y-4">
							<Label htmlFor="type-select" className="text-sm font-medium text-gray-700 mb-2 block">Type</Label>
							<Select value={type} onValueChange={(value) => {
								setType(value as 'model' | 'dataset')
								setSelectedDataset("")
								setSelectedModel("")
							}} key="type-select">
								<SelectTrigger className="bg-white border-gray-300 w-full">
									<SelectValue placeholder="Select Type" />
								</SelectTrigger>
								<SelectContent className="bg-white">
									<SelectItem value="model" className="hover:bg-gray-200">Model</SelectItem>
									<SelectItem value="dataset" className="hover:bg-gray-200">Dataset</SelectItem>
								</SelectContent>
							</Select>
							{type === 'dataset' && (
								<>
									<Label htmlFor="dataset-select" className="text-sm font-medium text-gray-700 mb-2 block">Dataset</Label>
									<Select value={selectedDataset} onValueChange={(value) => setSelectedDataset(value)} key="dataset-select">
										<SelectTrigger className="bg-white border-gray-300 w-full">
											<SelectValue placeholder="Select Dataset" />
										</SelectTrigger>
										<SelectContent className="bg-white">
											{datasets.map((ds) => (
												<SelectItem key={ds.id} value={ds.id.toString()} className="hover:bg-gray-200">{ds.name}</SelectItem>
											))}
										</SelectContent>
									</Select>
									<Button
										onClick={() => {
											// Handle dataset visualization
											visualizeDataset()
										}}
										disabled={loading || selectedDataset === ""}
										className="w-full bg-blue-600 hover:bg-blue-700 text-white"
									>
										{loading ? "Creating Visualizations..." : "Visualize Dataset"}
									</Button>
								</>
							)}
							{type === 'model' && (
								<>
									<Label htmlFor="model-select" className="text-sm font-medium text-gray-700 mb-2 block">Model</Label>
									<Select value={selectedModel} onValueChange={(value) => setSelectedModel(value)} key="model-select">
										<SelectTrigger className="bg-white border-gray-300 w-full">
											<SelectValue placeholder="Select Model" />
										</SelectTrigger>
										<SelectContent className="bg-white">
											{models.map((model) => (
												<SelectItem key={model.id} value={model.id.toString()} className="hover:bg-gray-200">{model.name}</SelectItem>
											))}
										</SelectContent>
									</Select>
									<Button
										onClick={() => {
											// Handle model evaluation and metric visualization
											evaluateModel()
										}}
										disabled={loading || selectedModel === ""}
										className="w-full bg-blue-600 hover:bg-blue-700 text-white"
									>
										{loading ? "Evaluating Model..." : "Evaluate Model"}
									</Button>
								</>
							)}
							{loading && <p>Loading...</p>}
						</div>
					</Card>

					{/* Experiment History */}
					<Card className="p-6">
						<div className="flex items-center justify-between mb-4">
							<h3 className="text-lg font-semibold">Recent Experiments</h3>
							{experimentData && (
								<Button
									variant="outline"
									size="sm"
									onClick={() => setExperimentData(null)}
									className="text-xs"
								>
									Clear View
								</Button>
							)}
						</div>
						
						{/* Search */}
						<input
							type="text"
							placeholder="Search experiments..."
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							className="w-full px-3 py-2 mb-3 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>

						{/* History List */}
						<div className="space-y-2 max-h-64 overflow-y-auto">
							{experimentHistory
								.filter((exp) => {
									if (!searchQuery.trim()) return true
									const query = searchQuery.toLowerCase()
									return (
										exp.data.model_name.toLowerCase().includes(query) ||
										exp.data.model_type.toLowerCase().includes(query) ||
										exp.data.type.toLowerCase().includes(query)
									)
								})
								.slice(0, showAllHistory ? undefined : 3)
								.map((exp) => {
									const date = new Date(exp.timestamp)
									const timeStr = date.toLocaleString('en-US', {
										month: 'short',
										day: 'numeric',
										hour: '2-digit',
										minute: '2-digit'
									})
									return (
										<div key={exp.id} className="p-3 bg-gray-50 hover:bg-blue-50 border border-gray-200 rounded transition-colors group">
											<button
												className="float-right opacity-0 group-hover:opacity-100 transition-opacity text-red-600 hover:text-red-700"
												title="Delete experiment"
												onClick={(e) => {
													e.stopPropagation()
													const ok = window.confirm('Delete this experiment from history? This cannot be undone.')
													if (!ok) return
													const updated = deleteExperimentFromHistory(exp.id)
													setExperimentHistory(updated)
												}}
											>
												<Trash2 className="w-4 h-4" />
											</button>
											<div onClick={() => setExperimentData(exp.data)} className="cursor-pointer">
												<div className="font-medium text-sm text-gray-900">{exp.data.model_name}</div>
												<div className="text-xs text-gray-600 mt-1">
													{exp.data.model_type} • {exp.data.type}
												</div>
												<div className="text-xs text-gray-500 mt-1">{timeStr}</div>
											</div>
										</div>
									)
								})}
							{experimentHistory.length === 0 && (
								<p className="text-sm text-gray-500 text-center py-4">No experiments yet</p>
							)}
						</div>

						{/* Show More/Less toggle */}
						{experimentHistory.length > 3 && (
							<Button
								variant="ghost"
								size="sm"
								onClick={() => setShowAllHistory(!showAllHistory)}
								className="w-full mt-2 text-xs text-blue-600 hover:text-blue-700"
							>
								{showAllHistory ? `Show Less` : `Show ${experimentHistory.length - 3} More`}
							</Button>
						)}
					</Card>
				</div>

				{/* Comparison Results */}
								<div className="lg:col-span-3">
										{experimentData ? (
												<div className="space-y-6">
														{/* Info Card */}
														<Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
																<CardHeader>
																		<CardTitle className="text-xl">{experimentData.model_name}</CardTitle>
																		<CardDescription>{experimentData.model_type} • {experimentData.type === 'classification' ? 'Classification' : 'Regression'}</CardDescription>
																</CardHeader>
														</Card>

														{/* Metrics Grid */}
														{experimentData.type === 'classification' ? (
															<div className="grid grid-cols-2 md:grid-cols-3 gap-4">
																{['accuracy','precision','recall','f1','roc_auc','pr_auc'].map((k) => (
																	experimentData.metrics[k] !== undefined && (
																		<Card key={k} className="p-4">
																			<CardTitle className="text-sm capitalize">{k.replace('_',' ').toUpperCase()}</CardTitle>
																			<CardContent className="pt-2">
																				<div className="text-2xl font-semibold">{(k==='accuracy'||k==='precision'||k==='recall'||k==='f1'||k==='roc_auc'||k==='pr_auc') ? Number(experimentData.metrics[k]).toFixed(k==='roc_auc'||k==='pr_auc'?3:3) : experimentData.metrics[k]}</div>
																			</CardContent>
																		</Card>
																	)
																))}
																{/* If multiclass APs are present but aggregate pr_auc isn't, show macro AP */}
																{experimentData.metrics['pr_auc'] === undefined && Array.isArray((experimentData.metrics as any).pr_auc_ovr) && (experimentData.metrics as any).pr_auc_ovr.length > 0 && (
																	(() => {
																		const aps = ((experimentData.metrics as any).pr_auc_ovr as Array<{class_label: string; ap: number}>).map(x => x.ap)
																		const macro = aps.reduce((a,b)=>a+b,0) / aps.length
																		return (
																			<Card className="p-4">
																				<CardTitle className="text-sm">PR AUC (Macro OvR)</CardTitle>
																				<CardContent className="pt-2">
																					<div className="text-2xl font-semibold">{macro.toFixed(3)}</div>
																				</CardContent>
																			</Card>
																		)
																	})()
																)}
															</div>
														) : (
																<div className="grid grid-cols-2 md:grid-cols-3 gap-4">
																{['mse','rmse','mae','r2'].map((k) => (
																	experimentData.metrics[k] !== undefined && (
																		<Card key={k} className="p-4">
																				<CardTitle className="text-sm uppercase">{k}</CardTitle>
																			<CardContent className="pt-2">
																					<div className="text-2xl font-semibold">{Number(experimentData.metrics[k]).toFixed(k==='r2'?3:4)}</div>
																			</CardContent>
																		</Card>
																	)
																))}
															</div>
														)}

														{/* Curves */}
														<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
															{/* Binary ROC */}
															{experimentData.metrics.roc_curve && Array.isArray(experimentData.metrics.roc_curve) && experimentData.metrics.roc_curve.length > 0 && (
																<Card>
																	<CardHeader>
																		<CardTitle className="text-lg">ROC Curve</CardTitle>
																		<CardDescription>AUC: {experimentData.metrics.roc_auc?.toFixed(3)}</CardDescription>
																	</CardHeader>
																	<CardContent>
																		<ResponsiveContainer width="100%" height={300}>
																			<LineChart data={experimentData.metrics.roc_curve as any} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
																				<CartesianGrid strokeDasharray="3 3" />
																				<XAxis dataKey="fpr" domain={[0,1]} type="number" label={{ value: "False Positive Rate", position: "insideBottomRight", offset: -5 }} />
																				<YAxis domain={[0,1]} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft" }} />
																				<Tooltip content={({ active, payload, label }: any) => {
																					if (!active || !payload || payload.length === 0) return null
																					const dataPoints = payload.filter((p: any) => p.name !== 'Baseline' && p.dataKey !== '__baseline')
																					if (dataPoints.length === 0) return null
																					const cursorFpr = typeof label === 'number' ? label : (dataPoints[0]?.payload?.fpr ?? 0)
																					return (
																						<div className="rounded-md bg-white/95 border border-gray-200 shadow p-2 text-xs">
																							<div className="font-medium text-gray-700 mb-1">FPR: {Number(cursorFpr).toFixed(3)}</div>
																							{dataPoints.map((point: any, idx: number) => (
																								<div key={idx} className="flex items-center gap-2">
																									<span className="inline-block w-2 h-2 rounded-sm" style={{ background: point.color }} />
																									<span className="text-gray-600">{point.name}:</span>
																									<span className="text-gray-900">TPR {Number(point.value).toFixed(3)}</span>
																								</div>
																							))}
																						</div>
																					)
																				}} />
																				<Legend />
																				{/* Baseline: random guessing diagonal */}
																				<Line
																					data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
																					type="linear"
																					dataKey="tpr"
																					stroke="#9ca3af"
																					strokeDasharray="4 4"
																					dot={false}
																					name="Baseline"
																					isAnimationActive={false}
																				/>
																				{/* ROC curve */}
																				<Line
																					type="monotone"
																					dataKey="tpr"
																					name="ROC"
																					stroke="#2563eb"
																					dot={false}
																					strokeWidth={2}
																					isAnimationActive={true}
																					activeDot={true as any}
																				/>
																			</LineChart>
																		</ResponsiveContainer>
																	</CardContent>
																</Card>
															)}

															{/* Multiclass ROC (OvR) overlay */}
															{Array.isArray((experimentData.metrics as any).roc_curves_ovr) && (experimentData.metrics as any).roc_curves_ovr.length > 0 && (
																(() => {
																	const rocCurvesOvr = (experimentData.metrics as any).roc_curves_ovr as Array<{class_label: string; curve: any[]}>
																	const mergedData = mergeRocCurves(rocCurvesOvr)
																	const palette = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c", "#0ea5e9", "#f59e0b", "#10b981"]
																	
																	return (
																		<Card>
																			<CardHeader>
																				<CardTitle className="text-lg">ROC Curves (OvR)</CardTitle>
																				<CardDescription>
																					One-vs-Rest ROC curves for each class
																				</CardDescription>
																			</CardHeader>
																			<CardContent>
																				<ResponsiveContainer width="100%" height={300}>
																					<LineChart data={mergedData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
																						<CartesianGrid strokeDasharray="3 3" />
																						<XAxis dataKey="fpr" domain={[0,1]} type="number" label={{ value: "False Positive Rate", position: "insideBottomRight", offset: -5 }} />
																						<YAxis domain={[0,1]} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft" }} />
																						<Tooltip content={({ active, payload, label }: any) => {
																							if (!active || !payload || payload.length === 0) return null
																							const dataPoints = payload.filter((p: any) => p.name !== 'Baseline' && p.dataKey !== '__baseline')
																							if (dataPoints.length === 0) return null
																							const cursorFpr = typeof label === 'number' ? label : (dataPoints[0]?.payload?.fpr ?? 0)
																							return (
																								<div className="rounded-md bg-white/95 border border-gray-200 shadow p-2 text-xs">
																									<div className="font-medium text-gray-700 mb-1">FPR: {Number(cursorFpr).toFixed(3)}</div>
																									{dataPoints.map((point: any, idx: number) => (
																										<div key={idx} className="flex items-center gap-2">
																											<span className="inline-block w-2 h-2 rounded-sm" style={{ background: point.color }} />
																											<span className="text-gray-600">{point.name}:</span>
																											<span className="text-gray-900">TPR {Number(point.value).toFixed(3)}</span>
																										</div>
																									))}
																								</div>
																							)
																						}} />
																						<Legend />
																						{/* Baseline: random guessing (non-interactive) */}
																						<Line
																							dataKey="__baseline"
																							data={mergedData.map(d => ({ ...d, __baseline: d.fpr }))}
																							type="linear"
																							stroke="#9ca3af"
																							strokeDasharray="4 4"
																							dot={false}
																							name="Baseline"
																							isAnimationActive={false}
																							activeDot={false as any}
																						/>
																						{rocCurvesOvr.map((cls, idx) => {
																							const color = palette[idx % palette.length]
																							return (
																								<Line
																									key={`roc-${cls.class_label}`}
																									type="monotone"
																									dataKey={cls.class_label}
																									name={cls.class_label}
																									stroke={color}
																									dot={false}
																									strokeWidth={2}
																								/>
																							)
																						})}
																					</LineChart>
																				</ResponsiveContainer>
																			</CardContent>
																		</Card>
																	)
																})()
															)}

															{experimentData.metrics.pr_curve && (
																<Card>
																	<CardHeader>
																		<CardTitle className="text-lg">Precision-Recall Curve</CardTitle>
																		<CardDescription>AP: {experimentData.metrics.pr_auc?.toFixed(3)}</CardDescription>
																	</CardHeader>
																	<CardContent>
																		<ResponsiveContainer width="100%" height={300}>
																			<LineChart data={experimentData.metrics.pr_curve as any} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
																				<CartesianGrid strokeDasharray="3 3" />
																				<XAxis dataKey="recall" domain={[0,1]} type="number" label={{ value: "Recall", position: "insideBottomRight", offset: -5 }} />
																				<YAxis domain={[0,1]} label={{ value: "Precision", angle: -90, position: "insideLeft" }} />
																				<Tooltip />
																				<Legend />
																				<Line type="monotone" dataKey="precision" name="PR" stroke="#16a34a" dot={false} strokeWidth={2} />
																			</LineChart>
																		</ResponsiveContainer>
																	</CardContent>
																</Card>
															)}

															{/* Multiclass PR (OvR) overlay - moved up above Confusion Matrix */}
															{Array.isArray((experimentData.metrics as any).pr_curves_ovr) && (experimentData.metrics as any).pr_curves_ovr.length > 0 && (
																(() => {
																	const prCurvesOvr = (experimentData.metrics as any).pr_curves_ovr as Array<{class_label: string; curve: any[]}>
																	const mergedData = mergePrCurves(prCurvesOvr)
																	const palette = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c", "#0ea5e9", "#f59e0b", "#10b981"]
																	return (
																		<Card>
																			<CardHeader>
																				<CardTitle className="text-lg">Precision-Recall Curves (OvR)</CardTitle>
																				<CardDescription>
																					{(() => {
																						const apList = ((experimentData.metrics as any).pr_auc_ovr || []) as Array<{class_label: string; ap: number}>
																						if (Array.isArray(apList) && apList.length > 0) {
																							const macro = apList.reduce((a, b) => a + (b.ap ?? 0), 0) / apList.length
																							return (
																								<div className="space-y-1">
																									<div>Macro AP: {macro.toFixed(3)}</div>
																									<div className="text-xs text-gray-600">
																										APs: {apList.map((x) => `${x.class_label}=${(x.ap ?? 0).toFixed(3)}`).join(', ')}
																									</div>
																							</div>
																						)
																					}
																					return 'One-vs-Rest PR curves for each class'
																				})()}
																			</CardDescription>
																		</CardHeader>
																		<CardContent>
																			<ResponsiveContainer width="100%" height={300}>
																				<LineChart data={mergedData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
																					<CartesianGrid strokeDasharray="3 3" />
																					<XAxis dataKey="recall" domain={[0,1]} type="number" label={{ value: "Recall", position: "insideBottomRight", offset: -5 }} />
																					<YAxis domain={[0,1]} label={{ value: "Precision", angle: -90, position: "insideLeft" }} />
																					<Tooltip content={<PrTooltip />} />
																					<Legend />
																					{prCurvesOvr.map((cls, idx) => {
																						const color = palette[idx % palette.length]
																						return (
																							<Line
																								key={`pr-${cls.class_label}`}
																								type="monotone"
																								dataKey={cls.class_label}
																								name={cls.class_label}
																								stroke={color}
																								dot={false}
																								strokeWidth={2}
																							/>
																						)
																					})}
																				</LineChart>
																			</ResponsiveContainer>
																		</CardContent>
																	</Card>
																)
																})()
															)}
														</div>

														{/* Confusion Matrix */}
														{experimentData.confusion_matrix && (
															<Card>
																<CardHeader>
																	<CardTitle className="text-lg">Confusion Matrix</CardTitle>
																	<CardDescription>Predicted vs Actual</CardDescription>
																</CardHeader>
																<CardContent>
																	<ConfusionMatrix labels={experimentData.confusion_matrix.labels} matrix={experimentData.confusion_matrix.matrix} />
																</CardContent>
															</Card>
														)}

												{/* SHAP Feature Importance */}
												{(experimentData as any)?.shap?.status === 'pending' && (
													<Card>
														<CardHeader>
															<CardTitle className="text-lg">Feature Importance (SHAP)</CardTitle>
															<CardDescription>Computing…</CardDescription>
														</CardHeader>
														<CardContent>
															<div className="w-full h-3 bg-gray-200 rounded">
																<div className="h-3 bg-purple-500 rounded" style={{ width: `${Math.min(100, Math.max(0, shapProgress))}%`, transition: 'width 0.5s ease' }} />
															</div>
															<p className="mt-2 text-xs text-gray-600">{Math.round(shapProgress)}% complete</p>
														</CardContent>
												</Card>
											)}

											{experimentData.shap && experimentData.shap.feature_importance && experimentData.shap.feature_importance.length > 0 && !(experimentData as any)?.shap?.status && (
														<Card>
															<CardHeader>
																	<CardTitle className="text-lg">Feature Importance (SHAP)</CardTitle>
																	<CardDescription>Mean |SHAP| values (top {Math.min(10, experimentData.shap.feature_importance.length)})</CardDescription>
																</CardHeader>
																<CardContent>
																	<ResponsiveContainer width="100%" height={Math.min(10, experimentData.shap.feature_importance.length) * 30 + 60}>
																		<BarChart data={[...experimentData.shap.feature_importance].slice(0,20).reverse()} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
																			<CartesianGrid strokeDasharray="3 3" />
																			<XAxis type="number" />
																			<YAxis type="category" dataKey="feature" width={100} />
																			<Tooltip />
																			<Legend />
																			<Bar dataKey="importance" name="Mean |SHAP|" fill="#8b5fbf" />
																		</BarChart>
																	</ResponsiveContainer>
																</CardContent>
															</Card>
														)}
												</div>
										) : (
						<Card className="p-12 text-center">
							<p className="text-gray-500">Select a model or dataset to visualize</p>
						</Card>
					)}
				</div>
			</div>
		</div>
	)
}

function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
	const size = labels.length
	const maxVal = Math.max(...matrix.flat()) || 1
	return (
		<div className="overflow-auto">
			<div className="inline-block">
				<div className="grid" style={{ gridTemplateColumns: `120px repeat(${size}, 80px)` }}>
					{/* Header Row */}
					<div></div>
					{labels.map((l) => (
						<div key={`h-${l}`} className="text-xs text-gray-600 text-center py-1">Pred {l}</div>
					))}
					{/* Rows */}
					{matrix.map((row, i) => (
						<>
							<div key={`rlabel-${i}`} className="text-xs text-gray-600 flex items-center">Actual {labels[i]}</div>
							{row.map((val, j) => {
								const intensity = val / maxVal
								const bg = `rgba(59,130,246,${0.15 + 0.65 * intensity})`
								return (
									<div
										key={`c-${i}-${j}`}
										className="text-sm font-medium text-center flex items-center justify-center border border-gray-200"
										style={{ width: 80, height: 48, background: bg, color: intensity > 0.6 ? 'white' : '#111827' }}
										title={`${labels[i]} vs ${labels[j]}: ${val}`}
									>
										{val}
									</div>
								)
							})}
						</>
					))}
				</div>
			</div>
		</div>
	)
}
