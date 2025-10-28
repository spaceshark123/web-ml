import { ArrowLeft, Search } from "lucide-react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DatasetCard } from "./dataset-card"
import { Link } from "react-router-dom"

export function DatasetsContent() {
  return (
    <div className="min-h-screen bg-gray-50">
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
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">Upload New Dataset</Button>
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
            <Input placeholder="Search datasets..." className="pl-10 bg-white border-gray-300" />
          </div>
          <Select defaultValue="all">
            <SelectTrigger className="w-48 bg-white border-gray-300">
              <SelectValue placeholder="All Types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="csv">CSV</SelectItem>
              <SelectItem value="json">JSON</SelectItem>
            </SelectContent>
          </Select>
          <Button className="bg-gray-700 hover:bg-gray-800 text-white">Filter</Button>
          <Button variant="outline" className="border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent">
            Clear
          </Button>
        </div>
      </div>

      {/* Datasets Grid */}
      <div className="px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <DatasetCard
            name="titanic"
            description="No description provided"
            uploadDate="Aug 23, 2025"
            fileSize="58.9 KB"
            rows={891}
            features={12}
            models={2}
          />
        </div>
      </div>
    </div>
  )
}
