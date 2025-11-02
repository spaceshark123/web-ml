import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Dataset {
  id: number
  name: string
  columns?: string[]
}

interface DatasetConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  dataset: Dataset | null
  onConfigSave: (config: { targetFeature: string; trainTestSplit: number }) => void
}

export function DatasetConfigDialog({ open, onOpenChange, dataset, onConfigSave }: DatasetConfigDialogProps) {
  const [targetFeature, setTargetFeature] = useState("")
  const [trainTestSplit, setTrainTestSplit] = useState("0.8")
  const [error, setError] = useState("")

  const handleSave = () => {
    if (!targetFeature) {
      setError("Please select a target feature")
      return
    }

    const splitValue = parseFloat(trainTestSplit)
    if (isNaN(splitValue) || splitValue <= 0 || splitValue >= 1) {
      setError("Train/test split must be a number between 0 and 1")
      return
    }

    onConfigSave({
      targetFeature,
      trainTestSplit: splitValue,
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Configure Dataset</DialogTitle>
          <DialogDescription>
            Select the target feature and set the train/test split ratio for your dataset.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="targetFeature">Target Feature</Label>
            <Select value={targetFeature} onValueChange={setTargetFeature}>
              <SelectTrigger>
                <SelectValue placeholder="Select target feature" />
              </SelectTrigger>
              <SelectContent>
                {dataset?.columns?.map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="trainTestSplit">Train/Test Split Ratio</Label>
            <Input
              id="trainTestSplit"
              type="number"
              value={trainTestSplit}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTrainTestSplit(e.target.value)}
              min="0"
              max="1"
              step="0.1"
              placeholder="0.8"
            />
            <p className="text-sm text-gray-500">Enter a value between 0 and 1 (e.g., 0.8 for 80% training data)</p>
          </div>
          {error && <p className="text-sm text-red-500">{error}</p>}
        </div>
        <div className="flex justify-end gap-3">
          <Button variant="outline" className="bg-gray-100 hover:bg-gray-200" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button className="bg-blue-600 hover:bg-blue-700 text-white" onClick={handleSave}>
            Save Configuration
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
