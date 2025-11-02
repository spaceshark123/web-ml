"use client"

import { useState, useEffect } from "react"
import { ArrowLeft } from "lucide-react"
import { Button } from "./ui/button"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "./ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Link } from "react-router-dom"
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

export function ExperimentsContent() {
	const [models, setModels] = useState<Model[]>([])
	const [datasets, setDatasets] = useState<Dataset[]>([])
	const [type, setType] = useState<'model' | 'dataset'>('model')
	const [selectedDataset, setSelectedDataset] = useState<string>("")
	const [selectedModel, setSelectedModel] = useState<string>("")
	const [experimentData, setExperimentData] = useState<ExperimentsResponse | null>(null)
	const [loading, setLoading] = useState(false)

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
				setLoading(true)
				try {
					const res = await fetch(`${API_BASE_URL}/models/${selectedModel}/experiments`, {
						credentials: 'include',
						headers: { 'Content-Type': 'application/json' },
					})
					const data: ExperimentsResponse = await res.json()
					setExperimentData(data)
				} catch (e) {
					console.error('Failed to evaluate model', e)
				} finally {
					setLoading(false)
				}
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
				<h1 className="text-4xl font-bold text-gray-900 mb-2">Experiments</h1>
				<p className="text-gray-500">Evaluate models, visualize datasets, and observe metrics</p>
			</div>

			<div className="p-8 grid grid-cols-1 lg:grid-cols-4 gap-8">
				{/* Selection */}
				<div className="lg:col-span-1">
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
															</div>
														) : (
															<div className="grid grid-cols-2 md:grid-cols-3 gap-4">
																{['mse','mae','r2'].map((k) => (
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
															{experimentData.metrics.roc_curve && (
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
																				<Tooltip />
																				<Legend />
																				<Line type="monotone" dataKey="tpr" name="ROC" stroke="#2563eb" dot={false} strokeWidth={2} />
																			</LineChart>
																		</ResponsiveContainer>
																	</CardContent>
																</Card>
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
														{experimentData.shap && experimentData.shap.feature_importance && experimentData.shap.feature_importance.length > 0 && (
															<Card>
																<CardHeader>
																	<CardTitle className="text-lg">Feature Importance (SHAP)</CardTitle>
																	<CardDescription>Mean |SHAP| values (top {Math.min(20, experimentData.shap.feature_importance.length)})</CardDescription>
																</CardHeader>
																<CardContent>
																	<ResponsiveContainer width="100%" height={320}>
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
