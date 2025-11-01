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
	const [maxDepth, setMaxDepth] = useState<number | "">("")
	const [criteria, setCriteria] = useState<"gini" | "entropy" | "">("")
	const [nEstimators, setNEstimators] = useState<number | "">("")

	const handleTrain = async () => {
		setError("")
	}

	/*
	<div className="space-y-2">
								<Label htmlFor="n-estimators">Number of Estimators</Label>
								<Input
									id="n-estimators"
									type="number"
									min={1}
									max={100}
									value={nEstimators}
									onChange={(e) => { setNEstimators(Number(e.target.value)) }}
									placeholder="Enter number of estimators (e.g., 10)"
								/>
							</div>
	*/

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
						// criteria and max_depth
						<>
							<div className="space-y-2">
								<Label htmlFor="criteria">Criteria</Label>
								<Select
									defaultValue={"gini"}
									onValueChange={(value) => {
										setCriteria(value as "gini" | "entropy")
									}}
								>
									<SelectTrigger className="w-full">
										<SelectValue placeholder="Select criteria" />
									</SelectTrigger>
									<SelectContent className="bg-white">
										<SelectItem value="gini" className="hover:bg-gray-200">Gini</SelectItem>
										<SelectItem value="entropy" className="hover:bg-gray-200">Entropy</SelectItem>
									</SelectContent>
								</Select>
							</div>
							<div className="space-y-2">
								<Label htmlFor="max-depth">Max Depth</Label>
								<Input
									id="max-depth"
									type="number"
									min={1}
									max={10}
									value={maxDepth}
									onChange={(e) => setMaxDepth(Number(e.target.value))}
									placeholder="Enter max depth (e.g., 5)"
								/>
							</div>
						</>
					)}
					{model.model_type === "random_forest" && (
						// n_estimators, criteria, max_depth
						<>
							<div className="space-y-2">
								<Label htmlFor="criteria">Criteria</Label>
								<Select
									defaultValue={"gini"}
									onValueChange={(value) => { setCriteria(value as "gini" | "entropy") }}
								>
									<SelectTrigger className="w-full">
										<SelectValue placeholder="Select criteria" />
									</SelectTrigger>
									<SelectContent className="bg-white">
										<SelectItem value="gini" className="hover:bg-gray-200">Gini</SelectItem>
										<SelectItem value="entropy" className="hover:bg-gray-200">Entropy</SelectItem>
									</SelectContent>
								</Select>
							</div>
							<div className="space-y-2">
								<Label htmlFor="max-depth">Max Depth</Label>
								<Input
									id="max-depth"
									type="number"
									min={1}
									max={10}
									value={maxDepth}
									onChange={(e) => { setMaxDepth(Number(e.target.value)) }}
									placeholder="Enter max depth (e.g., 5)"
								/>
							</div>
						</>
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
