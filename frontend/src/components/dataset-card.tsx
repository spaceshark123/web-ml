import { CreateModelDialog } from "./create-model-dialog"
import { Button } from "./ui/button"
import {
	Card,
	CardAction,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/components/ui/card"

interface DatasetCardProps {
	id: number
	name: string
	description: string
	uploadDate: string
	fileSize: string
	rows: number
	features: number
	target_feature: string
	models: number
	error?: string
	onDelete?: () => void
	onDownload?: () => void
}

export function DatasetCard({ id, name, description, uploadDate, fileSize, rows, target_feature, features, models, error, onDelete, onDownload }: DatasetCardProps) {
	return (
		<Card className="p-6 bg-white border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
			{/* Header */}
			<CardHeader>
				<CardTitle>{name}</CardTitle>
				<CardDescription>{description}</CardDescription>
			</CardHeader>

			{/* Dataset Metadata */}
			<CardContent>
				{error && <p className="text-red-500">{error}</p>}
				<div className="mb-6 space-y-2">
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Uploaded:</span>
						<span className="text-gray-900">{uploadDate}</span>
					</div>
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">File Size:</span>
						<span className="text-gray-900">{fileSize}</span>
					</div>
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Rows:</span>
						<span className="text-gray-900">{rows}</span>
					</div>
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Target:</span>
						<span className="text-gray-900">{target_feature}</span>
					</div>

				</div>

				{/* Stats */}
				<div className="grid grid-cols-3 gap-4 mb-6 pb-6 border-b border-gray-200">
					<div className="text-center">
						<div className="text-2xl font-bold text-blue-600">{rows}</div>
						<div className="text-xs text-gray-500 uppercase tracking-wide">Rows</div>
					</div>
					<div className="text-center">
						<div className="text-2xl font-bold text-blue-600">{features}</div>
						<div className="text-xs text-gray-500 uppercase tracking-wide">Features</div>
					</div>
					<div className="text-center">
						<div className="text-2xl font-bold text-blue-600">{models}</div>
						<div className="text-xs text-gray-500 uppercase tracking-wide">Models</div>
					</div>
				</div>

				{/* Action Buttons */}
				<div className="space-y-2">
					<div className="grid grid-cols-2 gap-2">
						<CreateModelDialog
							datasetIdInput={id}
							text="New Model"
							onUploadSuccess={() => {
								// Handle successful model creation
							}}
						/>
						<Button className="bg-green-600 hover:bg-green-700 text-white cursor-pointer">Train Model</Button>
					</div>
					<div className="grid grid-cols-2 gap-2">
						<Button variant='outline' className="border-gray-300 hover:bg-blue-50 bg-transparent text-blue-600 cursor-pointer">View</Button>
						<Button variant="outline" className="border-gray-300 text-gray-700 hover:bg-gray-100 bg-transparent cursor-pointer" onClick={onDownload}>
							Download
						</Button>
					</div>
					<Button variant="destructive" className="w-full bg-red-600 hover:bg-red-700 text-white cursor-pointer" onClick={onDelete}>
						Delete
					</Button>
				</div>
			</CardContent>
		</Card>
	)
}
