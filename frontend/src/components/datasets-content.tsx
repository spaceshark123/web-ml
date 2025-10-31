import { ArrowLeft, Search } from "lucide-react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DatasetCard } from "./dataset-card"
import { Link } from "react-router-dom"
import { useEffect, useState } from "react"
import { UploadDatasetDialog } from "./upload-dataset-dialog"
import { API_BASE_URL } from "@/constants"
import axios from "axios"

export interface Dataset {
	id: number
	name: string
	file_path: string
	upload_date: string
	file_size: number
	rows: number
	features: number
	target_feature: string
	models: number
	error?: string
	description?: string
}

const api = axios.create({
	baseURL: API_BASE_URL,
	withCredentials: true,
	headers: {
		'Content-Type': 'application/json',
		'Accept': 'application/json',
	},
});

export function DatasetsContent() {
	const [datasets, setDatasets] = useState<Dataset[]>([])
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState<string | null>(null)
	const [searchQuery, setSearchQuery] = useState("")
	const [selectedType, setSelectedType] = useState<string>("all")

	const fetchDatasets = async () => {
		setLoading(true)
		setError(null)
		try {
			const response = await fetch("http://localhost:5000/api/datasets", {
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
					throw new Error("Please log in to view your datasets")
				}
				throw new Error(`Server error: ${response.status}`)
			}

			const data = await response.json()
			setDatasets(data)
		} catch (error) {
			console.error("Failed to fetch datasets:", error)
			setError(error instanceof Error ? error.message : "Failed to load datasets. Please check if the backend server is running.")
		} finally {
			setLoading(false)
		}
	}

	const downloadDataset = async (datasetId: number, datasetName: string) => {
		try {
			const response = await fetch(`http://localhost:5000/api/download/${datasetId}`, {
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
			a.download = datasetName
			document.body.appendChild(a)
			a.click()
			a.remove()
			window.URL.revokeObjectURL(url)
		} catch (error) {
			console.error('Error downloading dataset:', error)
			alert('Failed to download dataset. Please try again.')
		}
	}

	const filteredDatasets = datasets.filter((dataset) => {
		const matchesSearch = dataset.name.toLowerCase().includes(searchQuery.toLowerCase());
		const matchesType = selectedType === "all" || dataset.file_path.endsWith(selectedType);
		return matchesSearch && matchesType;
	});

	useEffect(() => {
		fetchDatasets()
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
						<h1 className="text-4xl font-bold text-gray-900 mb-2">My Datasets</h1>
						<p className="text-gray-500">Manage and explore your uploaded datasets</p>
					</div>
					<div className="flex gap-3">
						<UploadDatasetDialog onUploadSuccess={fetchDatasets} />
					</div>
				</div>
			</div>

			{/* Search and Filter Section */}
			<div className="bg-white border-b border-gray-200 px-8 py-6">
				<div className="flex items-center gap-4">
					<div className="flex-1 relative">
						<Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
						<Input
							placeholder="Search datasets..."
							className="pl-10 bg-white border-gray-300"
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
						/>
					</div>
					<Select value={selectedType} onValueChange={(v) => setSelectedType(v)}>
						<SelectTrigger className="w-27 bg-white border-gray-300">
							<SelectValue placeholder="All Types" />
						</SelectTrigger>
						<SelectContent className="bg-white" >
							<SelectItem value="all" className="hover:bg-gray-200">All Types</SelectItem>
							<SelectItem value="csv" className="hover:bg-gray-200">CSV</SelectItem>
							<SelectItem value="xlsx" className="hover:bg-gray-200">XLSX</SelectItem>
							<SelectItem value="txt" className="hover:bg-gray-200">TXT</SelectItem>
							<SelectItem value="json" className="hover:bg-gray-200">JSON</SelectItem>
						</SelectContent>
					</Select>
					<Button
						variant="outline"
						className="border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent"
						onClick={() => {
							setSearchQuery("")
							setSelectedType("all")
						}}
					>
						Clear Filters
					</Button>
				</div>
			</div>

			{/* Datasets Grid */}
			<div className={`${loading ? 'filter blur-sm pointer-events-none select-none' : ''} px-8 py-8`}>
				{loading ? (
					<div className="text-center py-12">
						<p className="text-gray-500">Loading datasets...</p>
					</div>
				) : error ? (
					<div className="text-center py-12">
						<p className="text-red-500">{error}</p>
						<Button
							onClick={fetchDatasets}
							className="mt-4 bg-blue-600 hover:bg-blue-700 text-white"
						>
							Try Again
						</Button>
					</div>
				) : (
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
						{filteredDatasets.length === 0 ? (
								<div className="col-span-full text-center py-12">
									<p className="text-gray-500">No datasets found.</p>
								</div>
							) : (
								filteredDatasets.map((dataset) => {
								const handleDelete = async () => {
									const ok = window.confirm(`Delete dataset "${dataset.name}"? This will remove the file and cannot be undone.`)
									if (!ok) return
									try {
										const response = await fetch(`${API_BASE_URL}/datasets/${dataset.id}`, {
											method: 'DELETE',
											credentials: 'include',
											headers: {
												'Content-Type': 'application/json',
												'Authorization': `Bearer ${localStorage.getItem('token')}`
											}
										});

										if (!response.ok) {
											const data = await response.json().catch(() => ({}));
											const msg = data?.error || 'Failed to delete dataset';
											console.error('Delete failed', msg)
											alert(`Failed to delete dataset: ${msg}`)
											return
										}

										// Refresh the datasets list
										await fetchDatasets();
									} catch (error) {
										console.error('Error deleting dataset:', error);
										alert('Failed to delete dataset. See console for details.')
									}
								};

								return (
									<DatasetCard
										id={dataset.id}
										key={dataset.id}
										name={dataset.name}
										description={dataset.description || "No description provided"}
										uploadDate={dataset.upload_date ? new Date(dataset.upload_date).toLocaleDateString() : 'Unknown'}
										fileSize={dataset.file_size ? `${(dataset.file_size / 1024).toFixed(1)} KB` : 'Unknown'}
										rows={dataset.rows || 0}
										target_feature={dataset.target_feature || "N/A"}
										features={dataset.features || 0}
										models={dataset.models || 0}
										error={dataset.error}
										onDelete={handleDelete}
										onDownload={() => downloadDataset(dataset.id, dataset.name)}
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
							<div className="text-sm text-gray-600">Please wait — fetching datasets.</div>
						</div>
					</div>
				</div>
			)}
		</div>
	)
}
