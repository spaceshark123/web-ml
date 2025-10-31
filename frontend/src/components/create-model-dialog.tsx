import { use, useEffect, useState } from "react"
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

interface CreateModelDialogProps {
	text?: string
	onUploadSuccess?: () => void
}

export function CreateModelDialog({ text, onUploadSuccess }: CreateModelDialogProps) {
	const [open, setOpen] = useState(false)
	const [type, setType] = useState("")
	const [datasetId, setDatasetId] = useState(-1)
	const [customName, setCustomName] = useState("")
	const [uploading, setUploading] = useState(false)
	const [datasets, setDatasets] = useState<Array<{id: number, name: string}>>([])
	const [error, setError] = useState("")

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
			const response = await fetch("http://localhost:5000/api/models", {
				method: "POST",
				body: JSON.stringify({
					name: customName.trim(),
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
			onUploadSuccess?.()
			setOpen(false)
			setType("")
			setDatasetId(-1)
			setCustomName("")
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to create model. Please try again.")
		} finally {
			setUploading(false)
		}
	}

	const getAvailableDatasets = async () => {
		// Fetch available datasets from the backend to populate the select options
		const response = await fetch("http://localhost:5000/api/datasets", {
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

	useEffect(() => {
		if (open && datasets.length === 0) {
			getAvailableDatasets()
		}
	}, [open])

	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger asChild>
				<Button className="bg-blue-600 hover:bg-blue-700 text-white">{text ? text : "Create New Model"}</Button>
			</DialogTrigger>
			<DialogContent>
				<DialogHeader>
					<DialogTitle>Create New Model</DialogTitle>
					<DialogDescription>
						Fill in the details below to create a new model.
					</DialogDescription>
				</DialogHeader>
				<div className="space-y-4 py-4">
					<div className="space-y-2">
						<Label htmlFor="file">Target Dataset</Label>
						<Select
							onValueChange={(value) => setDatasetId(parseInt(value))}
						>
							<SelectTrigger className="w-full">
								<SelectValue placeholder="Select a dataset" />
							</SelectTrigger>
							<SelectContent>
								{datasets && datasets.map((dataset) => {
									return (
										<SelectItem key={dataset.id} value={dataset.id.toString()}>{dataset.name}</SelectItem>
									)
								})}
							</SelectContent>
						</Select>
					</div>
					<div className="space-y-2">
						<Label htmlFor="model-type">Model Type</Label>
						<Select
							onValueChange={(value) => setType(value)}
						>
							<SelectTrigger className="w-full">
								<SelectValue placeholder="Select model type" />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="linear_regression">Linear Regression</SelectItem>
								<SelectItem value="logistic_regression">Logistic Regression</SelectItem>
								<SelectItem value="decision_tree">Decision Tree</SelectItem>
								<SelectItem value="bagging">Bagging</SelectItem>
								<SelectItem value="boosting">Boosting</SelectItem>
								<SelectItem value="random_forest">Random Forest</SelectItem>
								<SelectItem value="svm">SVM</SelectItem>
								<SelectItem value="mlp">MLP</SelectItem>
							</SelectContent>
						</Select>
					</div>
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
					{error && <p className="text-sm text-red-500">{error}</p>}
				</div>
				<div className="flex justify-end gap-3">
					<Button variant="outline" className="bg-gray-100 hover:bg-gray-200" onClick={() => setOpen(false)}>
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
			</DialogContent>
		</Dialog>
	)
}
