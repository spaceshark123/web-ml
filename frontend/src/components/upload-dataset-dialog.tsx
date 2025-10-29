import { useState } from "react"
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

interface UploadDatasetDialogProps {
  onUploadSuccess?: () => void
}

export function UploadDatasetDialog({ onUploadSuccess }: UploadDatasetDialogProps) {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [customName, setCustomName] = useState("")
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      // Set default custom name from file name without extension
      const nameWithoutExt = selectedFile.name.replace(/\.[^/.]+$/, "")
      setCustomName(nameWithoutExt)
      setError("")
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file")
      return
    }

    setUploading(true)
    setError("")

    if (!customName.trim()) {
      setError("Please provide a dataset name")
      return
    }

    const formData = new FormData()
    formData.append("file", file)
    formData.append("name", customName.trim())

    try {
      const response = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
      })

      if (!response.ok) {
        throw new Error("Upload failed")
      }

      const data = await response.json()
      if (data.error) {
        throw new Error(data.error)
      }
      onUploadSuccess?.()
      setOpen(false)
      setFile(null)
      setCustomName("")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload dataset. Please try again.")
    } finally {
      setUploading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="bg-blue-600 hover:bg-blue-700 text-white">Upload New Dataset</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload Dataset</DialogTitle>
          <DialogDescription>
            Upload a CSV, XLSX, TXT, or JSON file containing your dataset. Make sure your data is properly formatted.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="file">Dataset File</Label>
            <Input
              id="file"
              type="file"
              accept=".csv,.txt,.xlsx"
              onChange={handleFileChange}
              className="cursor-pointer"
            />
          </div>
          {file && (
            <>
              <div className="space-y-2">
                <Label htmlFor="datasetName">Dataset Name</Label>
                <Input
                  id="datasetName"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  placeholder="Enter dataset name"
                />
              </div>
              <p className="text-sm text-gray-500">
                Selected file: {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
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
            onClick={handleUpload}
            disabled={!file || uploading}
          >
            {uploading ? "Uploading..." : "Upload"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
