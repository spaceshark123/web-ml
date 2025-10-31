import { ArrowLeft, Search } from "lucide-react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Link } from "react-router-dom"
import { useEffect, useState } from "react"
import { CreateModelDialog } from "./create-model-dialog"
import { API_BASE_URL } from "@/constants"
import axios from "axios"
import { ModelCard } from "./model-card"

export interface Model {
	id: number
	name: string
	created_at: string
	model_type: 'linear_regression' | 'logistic_regression' | 'decision_tree' | 'bagging' | 'boosting' | 'random_forest' | 'svm' | 'mlp'
	dataset_id: number
	metrics: {
		accuracy?: number
		precision?: number
		recall?: number
		f1_score?: number
		mse?: number
		rmse?: number
		mae?: number
		r2_score?: number
	}
	error?: string
}

const api = axios.create({
	baseURL: API_BASE_URL,
	withCredentials: true,
	headers: {
		'Content-Type': 'application/json',
		'Accept': 'application/json',
	},
});

export function ModelsContent() {
	const [models, setModels] = useState<Model[]>([])
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState<string | null>(null)

	const fetchModels = async () => {
		setLoading(true)
		setError(null)
		try {
			const response = await fetch("http://localhost:5000/api/models", {
				method: 'GET',
				credentials: 'include',
				headers: {
					'Accept': 'application/json',
					'Content-Type': 'application/json'
				},
				mode: 'cors'
			})

			if (!response.ok) {
				if (response.status === 401) {
					throw new Error("Please log in to view your models")
				}
				throw new Error(`Server error: ${response.status}`)
			}

			const data = await response.json()
			setModels(data)
		} catch (error) {
			console.error("Failed to fetch models:", error)
			setError(error instanceof Error ? error.message : "Failed to load models. Please check if the backend server is running.")
		} finally {
			setLoading(false)
		}
	}

	const downloadModel = async (modelId: number, modelName: string) => {
		try {
			const response = await fetch(`http://localhost:5000/api/models/${modelId}/download`, {
				method: 'GET',
				credentials: 'include',
			})
			if (!response.ok) {
				throw new Error('Download failed')
			}
			const blob = await response.blob()
			const url = window.URL.createObjectURL(blob)
			const a = document.createElement('a')
			a.href = url
			a.download = modelName
			document.body.appendChild(a)
			a.click()
			a.remove()
			window.URL.revokeObjectURL(url)
		} catch (error) {
			console.error('Error downloading dataset:', error)
			alert('Failed to download dataset. Please try again.')
		}
	}

	useEffect(() => {
		fetchModels()
	}, [])

	return (
		<div className="min-h-screen bg-gray-50 relative">
			{/* Back to Dashboard Link */}
			<div className="bg-white border-b border-gray-200 px-8 py-4">
				<Link to="/dashboard" className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 text-sm">
					<ArrowLeft className="w-4 h-4" />
					Back to Dashboard
				</Link>
			</div>

			{/* Header Section */}
			<div className="bg-white border-b border-gray-200 px-8 py-8">
				<div className="flex items-start justify-between">
					<div>
						<h1 className="text-4xl font-bold text-gray-900 mb-2">My Models</h1>
						<p className="text-gray-500">Manage and explore your models</p>
					</div>
					<div className="flex gap-3">
						<CreateModelDialog onUploadSuccess={fetchModels} />
						<Button variant="outline" className="border-gray-300 text-gray-700 hover:bg-gray-50 bg-transparent">
							Bulk Actions
						</Button>
					</div>
				</div>
			</div>

			{/* Search and Filter Section */}
			<div className="bg-white border-b border-gray-200 px-8 py-6">
				<div className="flex items-center gap-4">
					<div className="flex-1 relative">
						<Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
						<Input placeholder="Search models..." className="pl-10 bg-white border-gray-300" />
					</div>
					<Select defaultValue="all">
						<SelectTrigger className="w-27 bg-white border-gray-300">
							<SelectValue placeholder="All Types" />
						</SelectTrigger>
						<SelectContent className="bg-white" >
							<SelectItem value="all" className="hover:bg-gray-200">All Types</SelectItem>
							<SelectItem value="linear_regression" className="hover:bg-gray-200">Linear Regression</SelectItem>
							<SelectItem value="logistic_regression" className="hover:bg-gray-200">Logistic Regression</SelectItem>
							<SelectItem value="decision_tree" className="hover:bg-gray-200">Decision Tree</SelectItem>
							<SelectItem value="bagging" className="hover:bg-gray-200">Bagging</SelectItem>
							<SelectItem value="boosting" className="hover:bg-gray-200">Boosting</SelectItem>
							<SelectItem value="random_forest" className="hover:bg-gray-200">Random Forest</SelectItem>
							<SelectItem value="svm" className="hover:bg-gray-200">SVM</SelectItem>
							<SelectItem value="mlp" className="hover:bg-gray-200">MLP</SelectItem>
						</SelectContent>
					</Select>
					<Button className="bg-gray-700 hover:bg-gray-800 text-white">Filter</Button>
					<Button variant="outline" className="border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent">
						Clear
					</Button>
				</div>
			</div>

			{/* Models Grid */}
			<div className={`${loading ? 'filter blur-sm pointer-events-none select-none' : ''} px-8 py-8`}>
				{loading ? (
					<div className="text-center py-12">
						<p className="text-gray-500">Loading models...</p>
					</div>
				) : error ? (
					<div className="text-center py-12">
						<p className="text-red-500">{error}</p>
						<Button
							onClick={fetchModels}
							className="mt-4 bg-blue-600 hover:bg-blue-700 text-white"
						>
							Try Again
						</Button>
					</div>
				) : (
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
						{models.length === 0 ? (
							<div className="col-span-full text-center py-12">
								<p className="text-gray-500">No models found. Create your first model to get started!</p>
							</div>
						) : (
							models.map((model) => {
								const handleDelete = async () => {
									const ok = window.confirm(`Delete model "${model.name}"? This will remove the file and cannot be undone.`)
									if (!ok) return
									try {
										const response = await fetch(`${API_BASE_URL}/models/${model.id}`, {
											method: 'DELETE',
											credentials: 'include',
										});

										if (!response.ok) {
											const data = await response.json().catch(() => ({}));
											const msg = data?.error || 'Failed to delete model';
											console.error('Delete failed', msg)
											alert(`Failed to delete model: ${msg}`)
											return
										}

										// Refresh the models list
										await fetchModels();
									} catch (error) {
										console.error('Error deleting model:', error);
										alert('Failed to delete model. See console for details.')
									}
								};

								return (
									<ModelCard
										id={model.id}
										key={model.id}
										name={model.name}
										created_at={model.created_at ? new Date(model.created_at).toLocaleDateString() : 'Unknown'}
										model_type={model.model_type}
										datasetId={model.dataset_id}
										metrics={model.metrics}
										error={model.error}
										onDelete={handleDelete}
										onDownload={() => downloadModel(model.id, model.name)}
									/>
								);
							})
						)}
					</div>
				)}
			</div>

			{/* Loading overlay (blurred background visible behind) */}
			{loading && (
				<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
					<div className="backdrop-blur-sm absolute inset-0" aria-hidden="true" />
					<div className="relative z-10 bg-white/95 dark:bg-slate-800/95 p-6 rounded-lg shadow-lg flex items-center gap-4">
						<div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
						<div>
							<div className="font-semibold text-lg">Loading</div>
							<div className="text-sm text-gray-600">Please wait — fetching models.</div>
						</div>
					</div>
				</div>
			)}
		</div>
	)
}
