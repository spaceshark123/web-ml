import { Button } from "./ui/button"
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card"
import { useEffect, useState } from "react"
import { TrainModelDialog } from "./train-model-dialog"
import { useNavigate } from "react-router-dom"

interface ModelCardProps {
	id: number
	name: string
	description?: string
	model_type: 'linear_regression' | 'logistic_regression' | 'decision_tree' | 'bagging' | 'boosting' | 'random_forest' | 'svm' | 'mlp'
	created_at: string
	datasetId: number
	params?: Record<string, any> // params for model (e.g. n_estimators, layers, etc.)
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
	onDelete?: () => void
	onDownload?: () => void
	refreshModelsList?: () => void
}

export function ModelCard({ id, name, description, model_type, created_at, datasetId, params, metrics, error, onDelete, onDownload, refreshModelsList }: ModelCardProps) {
	const navigate = useNavigate()
	const datasetNameFromID = async (id: number) => {
		name = await fetch(`http://localhost:5000/api/datasets/${id}`, {
			method: 'GET',
			credentials: 'include',
			headers: {
				'Accept': 'application/json',
				'Content-Type': 'application/json'
			},
		})
			.then(response => response.json())
			.then(data => {
				return data.name;
			})
			.catch(error => {
				console.error("Failed to fetch dataset name:", error);
				return "Unknown Dataset";
			});
		return name;
	};

	const [datasetName, setDatasetName] = useState("Loading...");

	useEffect(() => {
		datasetNameFromID(datasetId).then(name => setDatasetName(name));
	}, []);

	return (
		<Card className="p-6 bg-white border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
			{/* Header */}
			<CardHeader>
				<CardTitle title={name} className="truncate">{name}</CardTitle>
				<CardDescription>{description}</CardDescription>
			</CardHeader>

			{/* Model Metadata */}
			<CardContent>
				{error && <p className="text-red-500">{error}</p>}
				<div className="mb-6 space-y-2">
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Created At:</span>
						<span className="text-gray-900">{created_at}</span>
					</div>
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Model Type:</span>
						<span className="text-gray-900">{model_type.replace('_', ' ').toUpperCase()}</span>
					</div>
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Dataset:</span>
						<span className="text-gray-900">{datasetName}</span>
					</div>
				</div>

				{/* Parameters */}
				{params && Object.keys(params).length > 0 && (
					<div className="mb-6">
						<h3 className="text-sm font-medium text-gray-700 mb-2">Parameters:</h3>
						<ul className="list-disc list-inside text-sm text-gray-600">
							{Object.entries(params).map(([key, value]) => (
								<li key={key}>
									<span className="font-semibold">{key.replace('_', ' ')}:</span> {String(value)}
								</li>
							))}
						</ul>
					</div>
				)}

				{/* Metrics */}
				<div className="grid grid-cols-3 gap-4 mb-6 pb-6 border-b border-gray-200">
					{metrics.accuracy !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{(metrics.accuracy * 100).toFixed(2)}%</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">Accuracy</div>
						</div>
					)}
					{metrics.precision !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{(metrics.precision * 100).toFixed(2)}%</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">Precision</div>
						</div>
					)}
					{metrics.recall !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{(metrics.recall * 100).toFixed(2)}%</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">Recall</div>
						</div>
					)}
					{metrics.f1_score !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{(metrics.f1_score * 100).toFixed(2)}%</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">F1 Score</div>
						</div>
					)}
					{metrics.mse !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{metrics.mse.toFixed(4)}</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">MSE</div>
						</div>
					)}
					{metrics.rmse !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{metrics.rmse.toFixed(4)}</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">RMSE</div>
						</div>
					)}
					{metrics.mae !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{metrics.mae.toFixed(4)}</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">MAE</div>
						</div>
					)}
					{metrics.r2_score !== undefined && (
						<div className="text-center">
							<div className="text-2xl font-bold text-blue-600">{(metrics.r2_score * 100).toFixed(2)}%</div>
							<div className="text-xs text-gray-500 uppercase tracking-wide">R² Score</div>
						</div>
					)}
				</div>

				{/* Action Buttons */}
				<div className="space-y-2">
					<div className="grid grid-cols-2 gap-2">
						<Button className="bg-blue-600 hover:bg-blue-700 text-white" onClick={() => navigate(`/experiments?model_id=${id}`)}>Evaluate</Button>
						<TrainModelDialog
							modelIdInput={id}
							onTrainSuccess={() => {
								// Handle successful train
								// refresh model list
								refreshModelsList?.()
							}}
						/>
					</div>
					<div className="grid grid-cols-2 gap-2">
						<Button variant="outline" className="border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent">
							View
						</Button>
						<Button variant="outline" className="border-gray-300 text-gray-700 hover:bg-gray-100 bg-transparent" onClick={onDownload}>
							Download
						</Button>
					</div>
					<Button variant="destructive" className="w-full bg-red-600 hover:bg-red-700 text-white" onClick={onDelete}>
						Delete
					</Button>
				</div>
			</CardContent>
		</Card>
	)
}
