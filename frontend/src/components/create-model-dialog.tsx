import { useEffect, useState } from "react"
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

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api"

interface CreateModelDialogProps {
	datasetIdInput?: number
	text?: string
	onCreateSuccess?: () => void
	refreshModelsList?: () => void
}

export function CreateModelDialog({ datasetIdInput, text, onCreateSuccess, refreshModelsList }: CreateModelDialogProps) {
	const [open, setOpen] = useState(false)
	const [config, setConfig] = useState(false) // flag for if config UI should be shown
	const [completed, setCompleted] = useState(false) // flag for if model creation is completed or if it should be deleted on close
	const [type, setType] = useState("")
	const [modelId, setModelId] = useState(-1)
	const [datasetId, setDatasetId] = useState(-1)
	const [customName, setCustomName] = useState("")
	const [description, setDescription] = useState("")
	const [uploading, setUploading] = useState(false)
	const [datasets, setDatasets] = useState<Array<{ id: number, name: string, regression: boolean }>>([])
	const [error, setError] = useState("")
	const [params, setParams] = useState<Record<string, any>>({}) // additional params for model

	const handleUpload = async () => {
		setUploading(true)
		setError("")

		if (datasetId === -1) {
			setError("Please select a dataset")
			return
		}

		if (type === "") {
			setError("Please select a model type")
			return
		}

		if (!customName.trim()) {
			setError("Please provide a model name")
			return
		}


		try {
			const response = await fetch(`${API_URL}/models`, {
				method: "POST",
				body: JSON.stringify({
					name: customName.trim(),
					description: description.trim(),
					model_type: type,
					dataset_id: datasetId,
				}),
				credentials: "include",
				headers: {
					'Accept': 'application/json',
					'Content-Type': 'application/json'
				},
			})

			if (!response.ok) {
				throw new Error("Failed to create model. Please try again.")
			}

			const data = await response.json()
			if (data.error) {
				throw new Error(data.error)
			}
			onCreateSuccess?.()
			setConfig(true)
			if (data.model_id) {
				setModelId(data.model_id)
			} else {
				throw new Error("Invalid response from server: missing model ID")
			}
			// setType("")
			// setDatasetId(-1)
			// setCustomName("")
			// setParams({})
			// setDescription("")
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to create model. Please try again.")
		} finally {
			setUploading(false)
		}
	}

	const handleSetParams = async () => {
		// validate params 
		setError("")
		if (!params || Object.keys(params).length === 0) {
			setError("Please provide model parameters")
			return
		}
		// if any one of params is -1, show error
		for (const key in params) {
			if (params[key] === -1) {
				setError("Please provide valid model parameters")
				return
			}
		}

		if (modelId === -1) {
			setError("Invalid model ID. Please try again.")
			return
		}

		// Send params to backend to update model
		try {
			const response = await fetch(`${API_URL}/models/${modelId}`, {
				method: "PUT",
				body: JSON.stringify({
					params: params,
				}),
				credentials: "include",
				headers: {
					'Accept': 'application/json',
					'Content-Type': 'application/json'
				},
			})

			if (!response.ok) {
				throw new Error("Failed to set model parameters. Please try again.")
			}

			const data = await response.json()
			if (data.error) {
				throw new Error(data.error)
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to set model parameters. Please try again.")
		}

		resetForm()
		onCreateSuccess?.()
	}

	const resetForm = () => {
		setType("")
		setDatasetId(-1)
		setCustomName("")
		setParams({})
		setDescription("")
		setError("")
		setConfig(false)
		setModelId(-1)
	}

	const getAvailableDatasets = async () => {
		// Fetch available datasets from the backend to populate the select options
		const response = await fetch(`${API_URL}/datasets`, {
			method: 'GET',
			credentials: 'include',
			headers: {
				'Accept': 'application/json',
				'Content-Type': 'application/json'
			},
		}).catch(error => {
			console.error("Failed to fetch datasets:", error);
			return null;
		});

		if (!response || !response.ok) {
			console.error("Failed to fetch datasets: Server error");
			return [];
		}
		const data = await response.json()
		setDatasets(data);
		return data
	}

	const handleDelete = async () => {
		try {
			const response = await fetch(`${API_URL}/models/${modelId}`, {
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
		} catch (error) {
			console.error('Error deleting model:', error);
			alert('Failed to delete model. See console for details.')
		}
	};

	useEffect(() => {
		if (open && datasets.length === 0) {
			getAvailableDatasets()
		}
		if (datasetIdInput) {
			setDatasetId(datasetIdInput)
		}
		if (!open) {
			setConfig(false)
			if (modelId !== -1 && !completed) {
				console.log("Deleting incomplete model with ID:", modelId)
				handleDelete().then(() => {
					refreshModelsList?.()
				})
			}
		}
		setCompleted(false)
	}, [open])

	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger asChild>
				<Button className="bg-blue-600 hover:bg-blue-700 text-white cursor-pointer">{text ? text : "Create New Model"}</Button>
			</DialogTrigger>
			<DialogContent>
				<DialogHeader>
					<DialogTitle>{config ? "Model Specifications" : "Create New Model"}</DialogTitle>
					<DialogDescription>
						{config ? "Review the specifications for the new model before creating it." : "Fill in the details below to create a new model."}
					</DialogDescription>
				</DialogHeader>
				{!config && (<>
					<div className="space-y-4 py-4">
						<div className="space-y-2">
							<Label htmlFor="custom-name">Model Name</Label>
							<Input
								id="custom-name"
								type="text"
								value={customName}
								onChange={(e) => setCustomName(e.target.value)}
								placeholder="Enter model name"
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="custom-description">Model Description</Label>
							<Input
								id="custom-description"
								type="text"
								value={description}
								onChange={(e) => setDescription(e.target.value)}
								placeholder="Enter model description"
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="file">Target Dataset</Label>
							<Select
								onValueChange={(value) => setDatasetId(parseInt(value))}
								defaultValue={datasetIdInput ? datasetIdInput.toString() : undefined}
							>
								<SelectTrigger className="w-full">
									<SelectValue placeholder="Select a dataset" />
								</SelectTrigger>
								<SelectContent className="bg-white">
									{datasets && datasets.map((dataset) => {
										return (
											<SelectItem key={dataset.id} value={dataset.id.toString()} className="hover:bg-gray-200">{dataset.name}</SelectItem>
										)
									})}
								</SelectContent>
							</Select>
						</div>
						<div className={`space-y-2 ${datasetId === -1 ? "opacity-50 pointer-events-none" : ""}`}>
							<Label htmlFor="model-type">Model Type</Label>
							<Select
								onValueChange={(value) => setType(value)}
								disabled={datasetId === -1}
							>
								<SelectTrigger className="w-full">
									<SelectValue placeholder="Select model type" />
								</SelectTrigger>
								<SelectContent className="bg-white">
									<SelectItem value="linear_regression" className="hover:bg-gray-200" disabled={
										// disable linear regression if selected dataset is not for regression
										!datasets.find(d => d.id === datasetId)?.regression
									}>Linear Regression</SelectItem>
									<SelectItem value="logistic_regression" className="hover:bg-gray-200" disabled={
										// disable logistic regression if selected dataset is not for classification
										datasets.find(d => d.id === datasetId)?.regression
									}>Logistic Regression</SelectItem>
									<SelectItem value="decision_tree" className="hover:bg-gray-200">Decision Tree</SelectItem>
									<SelectItem value="bagging" className="hover:bg-gray-200">Bagging</SelectItem>
									<SelectItem value="boosting" className="hover:bg-gray-200">Boosting</SelectItem>
									<SelectItem value="random_forest" className="hover:bg-gray-200">Random Forest</SelectItem>
									<SelectItem value="svm" className="hover:bg-gray-200">SVM</SelectItem>
									<SelectItem value="mlp" className="hover:bg-gray-200">MLP</SelectItem>
								</SelectContent>
							</Select>
						</div>
						{error && <p className="text-sm text-red-500">{error}</p>}
					</div>
					<div className="flex justify-end gap-3">
						<Button variant="outline" className="bg-gray-100 hover:bg-gray-200" onClick={() => { setOpen(false); resetForm(); }}>
							Cancel
						</Button>
						<Button
							className="bg-blue-600 hover:bg-blue-700 text-white"
							onClick={handleUpload}
							disabled={!type || datasetId === -1 || !customName.trim() || uploading}
						>
							{uploading ? "Uploading..." : "Upload"}
						</Button>
					</div>
				</>)}
				{config && (<>
					{type === "linear_regression" && (
						<div className="space-y-2">
							<p>Linear Regression has no additional parameters to configure.</p>
						</div>
					)}
					{type === "logistic_regression" && (
						<div className="space-y-2">
							<p>Logistic Regression has no additional parameters to configure.</p>
						</div>
					)}
					{(type === "decision_tree" || type === "random_forest") && (
						<div className="space-y-2">
							<Label htmlFor="criteria">Criteria</Label>
							<Select
								value={params.criterion || (() => {
									if (datasets.find(d => d.id === datasetId)?.regression) {
										setParams({ ...params, criterion: "squared_error" })
										return "squared_error"
									} else {
										setParams({ ...params, criterion: "gini" })
										return "gini"
									}
								})()}
								onValueChange={(value) => {
									setParams({ ...params, criterion: value })
								}}
							>
								<SelectTrigger className="w-full">
									<SelectValue placeholder="Select criteria" />
								</SelectTrigger>
								<SelectContent className="bg-white">
									<SelectItem value="gini" className="hover:bg-gray-200" disabled={datasets.find(d => d.id === datasetId)?.regression}>Gini</SelectItem>
									<SelectItem value="entropy" className="hover:bg-gray-200" disabled={datasets.find(d => d.id === datasetId)?.regression}>Entropy</SelectItem>
									<SelectItem value="squared_error" className="hover:bg-gray-200" disabled={!datasets.find(d => d.id === datasetId)?.regression}>MSE</SelectItem>
									<SelectItem value="absolute_error" className="hover:bg-gray-200" disabled={!datasets.find(d => d.id === datasetId)?.regression}>MAE</SelectItem>
								</SelectContent>
							</Select>
						</div>
					)}
					{(type === "decision_tree" || type === "random_forest") && (
						<div className="space-y-2">
							<Label htmlFor="max-depth">Max Depth</Label>
							<Input
								id="max-depth"
								type="number"
								min={1}
								max={10}
								value={params.max_depth || (() => {
									setParams({ ...params, max_depth: 5 })
									return 5
								})()}
								onChange={(e) => setParams({ ...params, max_depth: Number(e.target.value) })}
								placeholder="Enter max depth (e.g., 5)"
							/>
						</div>
					)}
					{(type === "random_forest" || type === "bagging" || type === "boosting") && (
						<div className="space-y-2">
							<Label htmlFor="n-estimators">Number of Estimators</Label>
							<Input
								id="n-estimators"
								type="number"
								min={1}
								max={100}
								value={params.n_estimators || (() => {
									setParams({ ...params, n_estimators: 10 })
									return 10
								})()}
								onChange={(e) => setParams({ ...params, n_estimators: Number(e.target.value) })}
								placeholder="Enter number of estimators (e.g., 10)"
							/>
						</div>
					)}
					{(type === "boosting") && (
						<div className="space-y-2">
							<Label htmlFor="learning-rate">Learning Rate</Label>
							<Input
								id="learning-rate"
								type="number"
								min={0.0001}
								max={5.0}
								step={0.001}
								value={params.learning_rate || (() => {
									if (type === "boosting") {
										setParams({ ...params, learning_rate: 1 })
										return 1
									} else {
										setParams({ ...params, learning_rate: 0.01 })
										return 0.01
									}
								})()}
								onChange={(e) => setParams({ ...params, learning_rate: Number(e.target.value) })}
								placeholder="Enter learning rate (e.g., 0.1)"
							/>
						</div>
					)}
					{(type === "svm") && (
						<div className="space-y-2">
							<Label htmlFor="kernel">Kernel</Label>
							<Select
								value={params.kernel || (() => {
									setParams({ ...params, kernel: "rbf" })
									return "rbf"
								})()}
								onValueChange={(value) => {
									setParams({ ...params, kernel: value })
								}}
							>
								<SelectTrigger className="w-full">
									<SelectValue placeholder="Select kernel" />
								</SelectTrigger>
								<SelectContent className="bg-white">
									<SelectItem value="linear" className="hover:bg-gray-200">Linear</SelectItem>
									<SelectItem value="poly" className="hover:bg-gray-200">Polynomial</SelectItem>
									<SelectItem value="rbf" className="hover:bg-gray-200">RBF</SelectItem>
									<SelectItem value="sigmoid" className="hover:bg-gray-200">Sigmoid</SelectItem>
								</SelectContent>
							</Select>
						</div>
					)}
					{error && <p className="text-sm text-red-500">{error}</p>}
				<div className="flex justify-end gap-3">
					<Button variant="outline" className="bg-gray-100 hover:bg-gray-200" onClick={() => {
						setOpen(false)
						handleDelete()
						refreshModelsList?.()
						resetForm()
					}}>
						Cancel
					</Button>
					<Button
						className="bg-blue-600 hover:bg-blue-700 text-white"
						onClick={() => {
							setCompleted(true)
							setOpen(false)
							handleSetParams()
							onCreateSuccess?.()
						}}
					>
						Create Model
					</Button>
				</div>
			</>)}
		</DialogContent>
		</Dialog >
	)
}
