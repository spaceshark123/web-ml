"use client"

import { useState, useEffect } from "react"
import { ArrowLeft, TrendingUp, TrendingDown, Minus } from "lucide-react"
import { Button } from "./ui/button"
import { Card, CardHeader, CardTitle, CardDescription } from "./ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Link } from "react-router-dom"
import { API_BASE_URL } from "@/constants"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"
import { Label } from "@/components/ui/label"
import type { Dataset } from "./datasets-content"
import type { Model } from "./models-content"

interface Data {
	type: 'model' | 'dataset'
	model_name?: string
	model_type?: string
	dataset_name?: string
	dataset_type?: string
	metrics: Record<string, any>
}

export function ExperimentsContent() {
	const [models, setModels] = useState<Model[]>([])
	const [datasets, setDatasets] = useState<Dataset[]>([])
	const [type, setType] = useState<'model' | 'dataset'>('model')
	const [selectedDataset, setSelectedDataset] = useState<string>("")
	const [selectedModel, setSelectedModel] = useState<string>("")
	const [experimentData, setExperimentData] = useState<Data | null>(null)
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
		// Simulate visualization process
		setTimeout(() => {
			setLoading(false)

			setExperimentData({
				type: 'dataset',
				dataset_name: datasets.find(ds => ds.id.toString() === selectedDataset)?.name || "Selected Dataset",
				dataset_type: datasets.find(ds => ds.id.toString() === selectedDataset)?.regression ? "Regression" : "Classification",
				metrics: {}
			})
		}, 2000)
	}

	const evaluateModel = async () => {
		setExperimentData(null)

		if (!selectedModel) return
		setLoading(true)
		// Simulate evaluation process
		setTimeout(() => {
			setLoading(false)

			setExperimentData({
				type: 'model',
				model_name: models.find(m => m.id.toString() === selectedModel)?.name || "Selected Model",
				model_type: models.find(m => m.id.toString() === selectedModel)?.model_type || "Model Type",
				metrics: {}
			})
		}, 2000)
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
									<CardTitle className="text-xl">{type === "model" ? experimentData.model_name : experimentData.dataset_name}</CardTitle>
									<CardDescription>{type === "model" ? experimentData.model_type : experimentData.dataset_type}</CardDescription>
								</CardHeader>
							</Card>

							{/* model: Classification Metrics */}

							{/* model: Regression Metrics */}

							{/* Preprocessing Info */}

							{/* Class Imbalance */}

							{/* Curves */}
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
