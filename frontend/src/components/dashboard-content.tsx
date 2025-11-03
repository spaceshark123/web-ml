import type React from "react"
import { useEffect, useState } from "react"
import { Database, Settings, FlaskConical, TrendingUp, Play, Eye } from "lucide-react"
import { Button } from "./ui/button"
import { Card } from "./ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"
import { Link, useNavigate } from "react-router-dom"
import { UploadDatasetDialog } from "./upload-dataset-dialog"
import type { Model } from "./models-content"
import type { Dataset } from "./datasets-content"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api"

// Import experiment history helpers
const EXPERIMENT_HISTORY_KEY = 'web-ml-experiment-history'
const COMPARISONS_COUNT_KEY = 'web-ml-comparisons-count'

interface SavedExperiment {
	id: string
	timestamp: number
	data: {
		model_id: number
		model_name: string
		model_type: string
		type: 'classification' | 'regression'
	}
}

function loadExperimentHistory(): SavedExperiment[] {
	try {
		const raw = localStorage.getItem(EXPERIMENT_HISTORY_KEY)
		return raw ? JSON.parse(raw) : []
	} catch {
		return []
	}
}

function loadComparisonsCount(): number {
	try {
		const raw = localStorage.getItem(COMPARISONS_COUNT_KEY)
		return raw ? parseInt(raw, 10) : 0
	} catch {
		return 0
	}
}

export function DashboardContent() {
	const [datasets, setDatasets] = useState<Dataset[]>([])
	const [models, setModels] = useState<Model[]>([])
	const [experiments, setExperiments] = useState<SavedExperiment[]>([])
	const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
	const navigate = useNavigate()

	const fetchDatasets = async () => {
		try {
			const res = await fetch(`${API_URL}/datasets`, {
				method: "GET",
				credentials: "include",
				headers: {
					Accept: "application/json",
					"Content-Type": "application/json",
				},
			})
			if (!res.ok) {
				return
			}
			const data = await res.json()
			if (data.error) {
				return
			}
			setDatasets(Array.isArray(data) ? data : [])
		} catch {
			console.log("Failed to fetch datasets")
		}
	}

	const fetchModels = async () => {
		try {
			const res = await fetch(`${API_URL}/models`, {
				method: "GET",
				credentials: "include",
				headers: {
					Accept: "application/json",
					"Content-Type": "application/json",
				},
			})
			if (!res.ok) {
				return
			}
			const data = await res.json()
			if (data.error) {
				return
			}
			setModels(Array.isArray(data) ? data : [])
		} catch {
			console.log("Failed to fetch models count")
		}
	}

	useEffect(() => {
		fetchDatasets()
		fetchModels()
		setExperiments(loadExperimentHistory())
	}, [])

	return (
		<div className="p-8 space-y-8">
			{/* Header */}
			<h1 className="text-4xl font-bold text-foreground">Dashboard</h1>

			{/* Stats Cards */}
			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
				<StatCard
					icon={<Database className="w-10 h-10" />}
					value={datasets.length.toString()}
					label="Datasets"
					gradient="from-[#6B7FD7] to-[#7B6FD7]"
				/>
				<StatCard
					icon={<Settings className="w-10 h-10" />}
					value={models.length.toString()}
					label="Models"
					gradient="from-[#7B6FD7] to-[#8B5FBF]"
				/>
				<StatCard
					icon={<FlaskConical className="w-10 h-10" />}
					value={experiments.length.toString()}
					label="Experiments Completed"
					gradient="from-[#8B5FBF] to-[#9B4FAF]"
				/>
				<StatCard
					icon={<TrendingUp className="w-10 h-10" />}
					value={loadComparisonsCount().toString()}
					label="Comparisons Done"
					gradient="from-[#9B4FAF] to-[#AB3F9F]"
				/>
			</div>

			{/* Test Model Performance */}
			<Card className="p-6 space-y-4 shadow-sm">
				<div className="flex items-center gap-2">
					<div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
						<Play className="w-3 h-3 text-white fill-white" />
					</div>
					<h2 className="text-xl font-semibold">Test Model Performance</h2>
				</div>

				<div className="space-y-4">
					<div>
						<label className="text-sm font-medium mb-2 block">Select a Model to Test:</label>
						<div className="flex gap-3">
							<Select onValueChange={(value) => setSelectedModelId(value)}>
								<SelectTrigger className="flex-1">
									<SelectValue placeholder="Choose a model..." />
								</SelectTrigger>
								<SelectContent className="bg-white">
									{models.map((model) => (
										<SelectItem key={model.id} value={model.id.toString()} className="hover:bg-gray-200">
											{model.name}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
							<Button className="bg-green-600 hover:bg-green-700 text-white" onClick={() => navigate(`/experiments?model_id=${selectedModelId}`) } disabled={!selectedModelId}>
								<Play className="w-4 h-4 mr-2 inline" />
								Run Performance Test
							</Button>
							<Button variant="outline" className="border-info text-info hover:bg-info/10 bg-transparent">
								<Link to="/models">
									<Eye className="w-4 h-4 mr-2 inline" />
									View Details
								</Link>
							</Button>
						</div>
					</div>
				</div>
			</Card>

			{/* Bottom Section */}
			<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
				{/* Recent Datasets */}
				<Card className="p-6 space-y-4 shadow-sm">
					<div className="flex items-center justify-between">
						<div className="flex items-center gap-2">
							<Database className="w-5 h-5 text-primary" />
							<h2 className="text-xl font-semibold">Recent Datasets</h2>
						</div>
						<UploadDatasetDialog text="+ Upload" onUploadSuccess={fetchDatasets} />
					</div>

					<div className="space-y-4">
						{datasets.slice(0, 3).map((dataset) => (
							<div key={dataset.id} className="flex items-start justify-between py-3 border-b max-w-full overflow-hidden last:border-0">
								<div className="max-w-[90%] overflow-hidden">
									<h3 className="font-semibold">{dataset.name}</h3>
									<p className="text-sm text-muted-foreground">{dataset.rows} rows, {dataset.features} columns</p>
								</div>
								<span className="text-sm text-muted-foreground">{new Date(dataset.upload_date).toLocaleDateString()}</span>
							</div>
						))}
					</div>

					<Button asChild variant="outline" className="w-full border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent">
						<Link to="/datasets">View All Datasets</Link>
					</Button>
				</Card>

				{/* Recent Models */}
				<Card className="p-6 space-y-4 shadow-sm">
					<div className="flex items-center gap-2">
						<Settings className="w-5 h-5 text-green-600" />
						<h2 className="text-xl font-semibold">Recent Models</h2>
					</div>

					<div className="space-y-3">
						{models.slice(0, 3).map((model) => (
							<div key={model.id} className="flex items-start justify-between py-3 border-b last:border-0 max-w-full overflow-hidden">
								<div className="flex-1 overflow-hidden max-w-[90%]">
									<div className="flex items-center gap-2 mb-1">
										<h3 className="font-semibold text-sm">{model.name}</h3>
										<span className="px-2 py-0.5 bg-gray-600 text-white text-xs rounded">{model.model_type}</span>
									</div>
									<p className="text-sm text-muted-foreground">Dataset ID: {model.dataset_id}</p>
								</div>
								<span className="text-sm text-muted-foreground whitespace-nowrap ml-4">{new Date(model.created_at).toLocaleDateString()}</span>
							</div>
						))}
					</div>

					<Button variant="outline" className="w-full border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent" asChild>
						<Link to="/models">View All Models</Link>
					</Button>
				</Card>

				{/* Recent Experiments */}
				<Card className="p-6 space-y-4 shadow-sm">
					<div className="flex items-center gap-2">
						<FlaskConical className="w-5 h-5 text-purple-600" />
						<h2 className="text-xl font-semibold">Recent Experiments</h2>
					</div>

					<div className="space-y-3">
						{experiments.slice(0, 3).map((exp) => {
							const date = new Date(exp.timestamp)
							const timeStr = date.toLocaleString('en-US', {
								month: 'short',
								day: 'numeric',
								hour: '2-digit',
								minute: '2-digit'
							})
							return (
								<div
									key={exp.id}
									onClick={() => navigate(`/experiments?id=${exp.id}`)}
									className="p-3 bg-gray-50 hover:bg-purple-50 border border-gray-200 rounded cursor-pointer transition-colors"
								>
									<div className="font-semibold text-sm text-gray-900">{exp.data.model_name}</div>
									<div className="text-xs text-gray-600 mt-1">
										{exp.data.model_type} • {exp.data.type}
									</div>
									<div className="text-xs text-gray-500 mt-1">{timeStr}</div>
								</div>
							)
						})}
						{experiments.length === 0 && (
							<p className="text-sm text-gray-500 text-center py-8">No experiments yet</p>
						)}
					</div>

					<Button variant="outline" className="w-full border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent" asChild>
						<Link to="/experiments">View All Experiments</Link>
					</Button>
				</Card>
			</div>
		</div>
	)
}

function StatCard({
	icon,
	value,
	label,
	gradient,
}: { icon: React.ReactNode; value: string; label: string; gradient: string }) {
	return (
		<Card className={`p-6 bg-gradient-to-br ${gradient} text-white shadow-lg border-0`}>
			<div className="flex flex-col items-center text-center space-y-3">
				{icon}
				<div className="text-4xl font-bold">{value}</div>
				<div className="text-sm font-medium opacity-90">{label}</div>
			</div>
		</Card>
	)
}
