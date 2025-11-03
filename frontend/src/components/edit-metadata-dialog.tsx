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

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api"

interface EditMetadataDialogProps {
  datasetId: number
  currentDataSource: string
  currentLicenseInfo: string
  onUpdateSuccess?: () => void
}

export function EditMetadataDialog({
  datasetId,
  currentDataSource,
  currentLicenseInfo,
  onUpdateSuccess,
}: EditMetadataDialogProps) {
  const [open, setOpen] = useState(false)
  const [dataSource, setDataSource] = useState(currentDataSource)
  const [licenseInfo, setLicenseInfo] = useState(currentLicenseInfo)
  const [updating, setUpdating] = useState(false)
  const [error, setError] = useState("")

  const handleUpdate = async () => {
    if (!dataSource.trim()) {
      setError("Data source is required")
      return
    }

    if (!licenseInfo.trim()) {
      setError("License information is required")
      return
    }

    setUpdating(true)
    setError("")

    try {
      const response = await fetch(`${API_URL}/datasets/${datasetId}`, {
        method: "PATCH",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          data_source: dataSource.trim(),
          license_info: licenseInfo.trim(),
        }),
      })

      let body: any = null
      const contentType = response.headers.get("content-type")
      if (contentType && contentType.includes("application/json")) {
        body = await response.json().catch(() => null)
      } else {
        body = await response.text().catch(() => null)
      }

      if (!response.ok) {
        const msg =
          body && typeof body === "object" && body.error
            ? body.error
            : typeof body === "string" && body
            ? body
            : `Update failed (${response.status})`
        throw new Error(msg)
      }

      // Success
      setOpen(false)
      onUpdateSuccess?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update metadata")
    } finally {
      setUpdating(false)
    }
  }

  const handleOpenChange = (isOpen: boolean) => {
    if (isOpen) {
      // Reset to current values when opening
      setDataSource(currentDataSource)
      setLicenseInfo(currentLicenseInfo)
      setError("")
    }
    setOpen(isOpen)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="text-xs border-gray-300 hover:bg-gray-100 cursor-pointer"
        >
          Edit Metadata
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit Dataset Metadata</DialogTitle>
          <DialogDescription>
            Update the data source and license information. Both fields are required.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="editDataSource">Data Source *</Label>
            <Input
              id="editDataSource"
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              placeholder="e.g., Kaggle URL, internal table, etc."
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="editLicenseInfo">License *</Label>
            <Input
              id="editLicenseInfo"
              value={licenseInfo}
              onChange={(e) => setLicenseInfo(e.target.value)}
              placeholder="e.g., CC BY 4.0, MIT, Proprietary"
              required
            />
          </div>
          {error && <p className="text-sm text-red-500">{error}</p>}
        </div>
        <div className="flex justify-end gap-3">
          <Button
            variant="outline"
            className="bg-gray-100 hover:bg-gray-200 cursor-pointer"
            onClick={() => setOpen(false)}
            disabled={updating}
          >
            Cancel
          </Button>
          <Button
            className="bg-blue-600 hover:bg-blue-700 text-white cursor-pointer"
            onClick={handleUpdate}
            disabled={updating}
          >
            {updating ? "Updating..." : "Update"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
