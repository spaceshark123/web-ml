import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  MultiSelect,
  MultiSelectContent,
  MultiSelectGroup,
  MultiSelectItem,
  MultiSelectTrigger,
  MultiSelectValue,
} from "@/components/ui/multi-select"
import { Input } from "@/components/ui/input"
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"

interface UploadDatasetDialogProps {
  text?: string
  onUploadSuccess?: () => void
}

export function UploadDatasetDialog({ text, onUploadSuccess }: UploadDatasetDialogProps) {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [customName, setCustomName] = useState("")
  const [description, setDescription] = useState("")
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState("")
  const [step, setStep] = useState<'select' | 'specify'>('select')
  const [columns, setColumns] = useState<string[] | null>(null)
  const [inputFeatures, setInputFeatures] = useState<string[] | null>(null)
  const [targetVariable, setTargetVariable] = useState<string | null>(null)
  const [splitPercent, setSplitPercent] = useState<number>(20)
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [preprocessing, setPreprocessing] = useState(false)
  const MAX_BYTES = 200 * 1024 * 1024 // 200 MB
  const uploadAbortCtrlRef = useRef<AbortController | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.size > MAX_BYTES) {
        setError('File is too large. Maximum allowed size is 200 MB.')
        setFile(null)
        return
      }
      setFile(selectedFile)
      // Set default custom name from file name without extension
      const nameWithoutExt = selectedFile.name.replace(/\.[^/.]+$/, "")
      setCustomName(nameWithoutExt)
      setError("")

      // Try to read header row for CSV/TXT to populate columns for specification step
      const ext = selectedFile.name.split('.').pop()?.toLowerCase()
      if (ext === 'csv' || ext === 'txt') {
        const reader = new FileReader()
        const CHUNK = 100 * 1024 // 100KB
        const blob = selectedFile.slice(0, CHUNK)
        reader.onload = () => {
          const text = reader.result as string
          if (!text) {
            setColumns(null)
            return
          }
          const lines = text.split(/\r?\n/).filter(Boolean)
          if (lines.length === 0) {
            setColumns(null)
            return
          }
          const header = lines[0]
          let cols: string[] = []
          try {
            if (ext === 'csv') {
              // basic CSV split, handles simple cases
              cols = header.split(',').map((s) => s.replace(/^"|"$/g, '').trim())
            } else {
              // txt assumed tab-separated
              cols = header.split('\t').map((s) => s.replace(/^"|"$/g, '').trim())
            }
            setColumns(cols.filter(Boolean))
          } catch (e) {
            setColumns(null)
          }
        }
        reader.onerror = () => setColumns(null)
        reader.readAsText(blob)
      } else {
        // For non-text files (xlsx), we cannot parse here—specification will need manual input
        setColumns(null)
      }
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
      setUploading(false)
      return
    }

    const formData = new FormData()
    formData.append("file", file)
    formData.append("name", customName.trim())
    if (description.trim()) {
      formData.append("description", description.trim())
    }

    try {
      const controller = new AbortController()
      uploadAbortCtrlRef.current = controller
      const response = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
        mode: 'cors',
        signal: controller.signal,
      })

      // try to parse server response (JSON) for a clearer error message
      let body: any = null
      const contentType = response.headers.get('content-type')
      if (contentType && contentType.includes('application/json')) {
        body = await response.json().catch(() => null)
      } else {
        body = await response.text().catch(() => null)
      }

      if (!response.ok) {
        const msg = body && typeof body === 'object' && body.error ? body.error : (typeof body === 'string' && body) || `Upload failed (${response.status})`
        throw new Error(msg)
      }

      const data = body
      if (data && data.error) {
        throw new Error(data.error)
      }

      onUploadSuccess?.()
      // proceed to specification step instead of closing
      // set dataset id from server response so we can persist specs
      if (data && data.dataset && data.dataset.id) {
        setDatasetId(data.dataset.id)
        // fetch columns from backend (more accurate than client-side header read)
        try {
          const colsRes = await fetch(`http://localhost:5000/api/datasets/${data.dataset.id}/columns`, {
            method: 'GET',
            credentials: 'include',
          })
          if (colsRes.ok) {
            const colsBody = await colsRes.json()
            if (colsBody && Array.isArray(colsBody.columns)) {
              setColumns(colsBody.columns)
            }
          }
        } catch (e) {
          // ignore - columns may be parsed locally or entered manually
        }
      }
      setStep('specify')
      // keep the file and customName so user can complete specs
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload dataset. Please try again.")
    } finally {
      setUploading(false)
      uploadAbortCtrlRef.current = null
    }
  }

  const performDeleteDataset = async (id: number) => {
    try {
      await fetch(`http://localhost:5000/api/datasets/${id}`, {
        method: 'DELETE',
        credentials: 'include',
      })
    } catch (e) {
      // best-effort; ignore errors
    }
  }

  const handleCancel = async () => {
    // If an upload is in progress, abort it
    if (uploading && uploadAbortCtrlRef.current) {
      try {
        uploadAbortCtrlRef.current.abort()
      } catch (_) { }
    }

    // If we have an uploaded dataset but specs not saved, delete it
    if (datasetId) {
      await performDeleteDataset(datasetId)
      // notify user and refresh dataset listing
      try {
        alert('Specification incomplete – Dataset upload cancelled')
        onUploadSuccess?.() // refresh the parent's dataset listing
      } catch (_) { }
      setDatasetId(null)
    }

    // Reset UI state and close
    setStep('select')
    setFile(null)
    setCustomName('')
    setDescription('')
    setColumns(null)
    setTargetVariable(null)
    setSplitPercent(20)
    setError('')
    setOpen(false)
  }

  // Ensure dataset is deleted if user reloads or navigates away before completing specs
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if ((step === 'specify' && datasetId) || uploading) {
        // Try to abort upload
        try { uploadAbortCtrlRef.current?.abort() } catch (_) { }
        // Attempt to delete dataset using keepalive so browser may send it
        if (datasetId) {
          try {
            // fetch with keepalive; best-effort
            fetch(`http://localhost:5000/api/datasets/${datasetId}`, { method: 'DELETE', credentials: 'include', keepalive: true })
          } catch (_) { }
        }
        e.preventDefault()
        e.returnValue = ''
        return ''
      }
      return undefined
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [datasetId, step, uploading])

  const handleSubmitSpecs = async () => {
    // Basic validation
    if (!targetVariable) {
      setError('Please select a target variable')
      return
    }
    if (!columns || !columns.includes(targetVariable)) {
      setError('Target variable must be one of the dataset columns')
      return
    }
    if (!inputFeatures || inputFeatures.length === 0) {
      setError('Please select at least one input feature')
      return
    }
    if (splitPercent <= 0 || splitPercent >= 100) {
      setError('Test split must be between 0 and 100')
      return
    }

    try {
      if (!datasetId) {
        setError('Unable to persist specifications: missing dataset id')
        return
      }

      const res = await fetch(`http://localhost:5000/api/datasets/${datasetId}/config`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_features: inputFeatures.join(","), target_feature: targetVariable, train_test_split: splitPercent }),
      })

      let body: any = null
      const ct = res.headers.get('content-type')
      if (ct && ct.includes('application/json')) {
        body = await res.json().catch(() => null)
      } else {
        body = await res.text().catch(() => null)
      }

      if (!res.ok) {
        const msg = body && typeof body === 'object' && body.error ? body.error : (typeof body === 'string' && body) || `Failed to save specs (${res.status})`
        throw new Error(msg)
      }
      setPreprocessing(true)
      const preprocessingRes = await fetch(`http://localhost:5000/api/datasets/${datasetId}/preprocess`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      })
      let preprocessBody: any = null
      const preprocessCt = preprocessingRes.headers.get('content-type')
      if (preprocessCt && preprocessCt.includes('application/json')) {
        preprocessBody = await preprocessingRes.json().catch(() => null)
      } else {
        preprocessBody = await preprocessingRes.text().catch(() => null)
      }

      if (!preprocessingRes.ok) {
        const msg = preprocessBody && typeof preprocessBody === 'object' && preprocessBody.error ? preprocessBody.error : (typeof preprocessBody === 'string' && preprocessBody) || `Failed to preprocess dataset (${preprocessingRes.status})`
        throw new Error(msg)
      }

      // success - close and reset
      setOpen(false)
      setStep('select')
      setFile(null)
      setCustomName('')
      setDescription('')
      setColumns(null)
      setInputFeatures(null)
      setTargetVariable(null)
      setSplitPercent(20)
      setDatasetId(null)
      setError('')
      onUploadSuccess?.()
      setPreprocessing(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save specifications')
    }
  }

  const handleDialogOpenChange = async (isOpen: boolean) => {
    // If closing while on specification step, delete uploaded dataset if it wasn't saved
    if (!isOpen && step === 'specify') {
      // if a dataset was created on the server but user didn't finish/save specs, delete it
      if (datasetId) {
        try {
          await fetch(`http://localhost:5000/api/datasets/${datasetId}`, {
            method: 'DELETE',
            credentials: 'include',
          })
        } catch (e) {
          // ignore network errors — deletion is best-effort
        }
        // Notify user about cancellation
        try {
          // Use a simple alert so we don't depend on a toast system which may not be available
          alert('Specification incomplete -- Dataset upload cancelled')
        } catch (_) { }
        onUploadSuccess?.() // refresh the parent's dataset listing
      }

      // reset to initial selection state
      setStep('select')
      setFile(null)
      setCustomName('')
      setDescription('')
      setColumns(null)
      setInputFeatures(null)
      setTargetVariable(null)
      setSplitPercent(20)
      setDatasetId(null)
      setError('')
    }
    setOpen(isOpen)
  }

  return (
    <Dialog open={open} onOpenChange={handleDialogOpenChange}>
      <DialogTrigger asChild>
        <Button className="bg-blue-600 hover:bg-blue-700 text-white">{text ? text : "Upload New Dataset"}</Button>
      </DialogTrigger>
      <DialogContent>
        {step === 'select' ? (
          <>
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
                  <div className="space-y-2">
                    <Label htmlFor="datasetDescription">Description (Optional)</Label>
                    <Input
                      id="datasetDescription"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Enter dataset description"
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
              <Button variant="outline" className="bg-gray-100 hover:bg-gray-200 cursor-pointer" onClick={handleCancel}>
                Cancel
              </Button>
              <Button
                className="bg-blue-600 hover:bg-blue-700 text-white cursor-pointer"
                onClick={handleUpload}
                disabled={!file || uploading}
              >
                {uploading ? "Uploading..." : "Upload"}
              </Button>
            </div>
          </>
        ) : (
          // Specification step
          <>
            <DialogHeader>
              <DialogTitle>Dataset Specifications</DialogTitle>
              <DialogDescription>
                Choose the input features, target variable, and the train/test split for downstream tasks.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Input Features</Label>
                {columns && columns.length > 0 ? (
                  <div>
                    <MultiSelect
                      values={inputFeatures || []}
                      onValuesChange={(values) => setInputFeatures(values)}
                    >
                      <MultiSelectTrigger className="w-full cursor-pointer">
                        <MultiSelectValue placeholder="Select input features" />
                      </MultiSelectTrigger>
                      <MultiSelectContent className="bg-white">
                        <MultiSelectGroup>
                          {columns.map((c) => (
                            <MultiSelectItem key={c} value={c} className="hover:bg-gray-200 cursor-pointer">{c}</MultiSelectItem>
                          ))}
                        </MultiSelectGroup>
                      </MultiSelectContent>
                    </MultiSelect>
                    {inputFeatures && inputFeatures.length > 0 && (
                      <div className="mt-2 text-sm">Selected: <strong>{inputFeatures.join(', ')}</strong></div>
                    )}
                  </div>
                ) : (
                  <div>
                    <Input
                      placeholder="Enter input feature names, separated by commas (unable to auto-detect columns)"
                      value={inputFeatures ? inputFeatures.join(', ') : ''}
                      onChange={(e) => setInputFeatures(e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                    />
                    <p className="text-sm text-gray-500 mt-1">Note: Column auto-detection is not available for this file type.</p>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label>Target Variable</Label>
                {columns && columns.length > 0 ? (
                  <div>
                    <Select value={targetVariable ?? undefined} onValueChange={(v) => setTargetVariable(v)}>
                      <SelectTrigger className="w-full cursor-pointer">
                        <SelectValue placeholder="Select target variable" />
                      </SelectTrigger>
                      <SelectContent className="bg-white">
                        {columns.map((c) => (
                          <SelectItem key={c} value={c} className="hover:bg-gray-200 cursor-pointer">{c}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {targetVariable && <div className="mt-2 text-sm">Selected: <strong>{targetVariable}</strong></div>}
                  </div>
                ) : (
                  <div>
                    <Input
                      placeholder="Enter target variable name (unable to auto-detect columns)"
                      value={targetVariable ?? ''}
                      onChange={(e) => setTargetVariable(e.target.value)}
                    />
                    <p className="text-sm text-gray-500 mt-1">Note: Column auto-detection is not available for this file type.</p>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label>Test Split (%)</Label>
                <div className="flex items-center gap-3">
                  <Input
                    type="number"
                    min={1}
                    max={99}
                    value={splitPercent}
                    onChange={(e) => setSplitPercent(Number(e.target.value))}
                    className="w-28"
                  />
                  <div className="text-sm text-gray-500">percent of data used for testing (1-99)</div>
                </div>
              </div>

              {error && <p className="text-sm text-red-500">{error}</p>}
            </div>
            <div className="flex justify-end gap-3">
              <Button variant="outline" className="bg-gray-100 hover:bg-gray-200 cursor-pointer" onClick={handleCancel} disabled={preprocessing}>
                Cancel
              </Button>
              <Button className="bg-blue-600 hover:bg-blue-700 text-white cursor-pointer" onClick={handleSubmitSpecs} disabled={preprocessing}>
                Save Specifications
              </Button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}