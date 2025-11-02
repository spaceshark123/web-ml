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
import type { Model } from "./models-content"

interface TrainModelDialogProps {
	modelIdInput?: number
	text?: string
	onTrainSuccess?: () => void
}

export function TrainModelDialog({ modelIdInput, text, onTrainSuccess }: TrainModelDialogProps) {
	const [open, setOpen] = useState(false)
	const [model, setModel] = useState({ id: -1, name: "Model not found", created_at: "", model_type: "linear_regression", dataset_id: -1, metrics: {} } as Model)
	const [error, setError] = useState("")

	// hyperparameters state
	const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({})

	const handleTrain = async () => {
		setError("")
		try {
			const body: Record<string, any> = {}
			if (Object.keys(hyperparameters).length > 0) {
				body.hyperparams = hyperparameters
			}
			const response = await fetch(`http://localhost:5000/api/train/${model.id}`, {
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
					return
				} else {
					setError(`Server error: ${response.status}`)
					return
				}
			}
			const data = await response.json()
			if (data.error) {
				setError(data.error)
				return
			}
			// log
			console.log("Train " + model.name + " response:", data)
			alert("Training started successfully!")
			setOpen(false)

			// success
			onTrainSuccess?.()
		} catch (error) {
			setError("Failed to start training. Please try again.")
			return
		}
	}

	const getModel = async () => {
		try {
			const response = await fetch(`http://localhost:5000/api/models/${modelIdInput}`, {
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
		} catch {
			console.log("Failed to fetch model")
		}
	}

	useEffect(() => {
		if (open && modelIdInput) {
			getModel()
		}
	}, [open, modelIdInput])

	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger asChild>
				<Button className="bg-green-600 hover:bg-green-700 text-white">{text ? text : "Train Model"}</Button>
			</DialogTrigger>
			<DialogContent>
				<DialogHeader>
					<DialogTitle>Train {model.name}</DialogTitle>
					<DialogDescription>
						Enter the hyperparameters below to train the model.
					</DialogDescription>
				</DialogHeader>
				<div className="space-y-4 py-4">
					{model.model_type === "linear_regression" && (
						// no hyperparameters for linear regression
						<p className="text-sm text-gray-600">No hyperparameters to set for Linear Regression.</p>
					)}
					{model.model_type === "logistic_regression" && (
						// no hyperparameters for logistic regression
						<p className="text-sm text-gray-600">No hyperparameters to set for Logistic Regression.</p>
					)}
					{model.model_type === "decision_tree" && (
						// no hyperparameters for decision tree
						<p className="text-sm text-gray-600">No hyperparameters to set for Decision Tree.</p>
					)}
					{model.model_type === "random_forest" && (
						// no hyperparameters for random forest
						<p className="text-sm text-gray-600">No hyperparameters to set for Random Forest.</p>
					)}
					{model.model_type === "bagging" && (
						// no hyperparameters for bagging
						<p className="text-sm text-gray-600">No hyperparameters to set for Bagging.</p>
					)}
					{model.model_type === "boosting" && (
						// no hyperparameters for boosting
						<p className="text-sm text-gray-600">No hyperparameters to set for Boosting.</p>
					)}
					{model.model_type === "svm" && (
						// no hyperparameters for SVM
						<p className="text-sm text-gray-600">No hyperparameters to set for SVM.</p>
					)}
					{model.model_type === "mlp" && (
						// no hyperparameters for MLP
						<p className="text-sm text-gray-600">No hyperparameters to set for MLP.</p>
					)}
					{error && <p className="text-sm text-red-500">{error}</p>}
				</div>
				<div className="flex justify-end gap-3">
					<Button variant="outline" className="bg-gray-100 hover:bg-gray-200" onClick={() => setOpen(false)}>
						Cancel
					</Button>
					<Button
						className="bg-blue-600 hover:bg-blue-700 text-white"
						onClick={handleTrain}
						disabled={model.id === -1}
					>
						Train
					</Button>
				</div>
			</DialogContent>
		</Dialog>
	)
}
