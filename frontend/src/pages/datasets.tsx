import { Sidebar } from "@/components/sidebar"
import { DatasetsContent } from "@/components/datasets-content"

export default function DatasetsPage() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1">
        <DatasetsContent />
      </main>
    </div>
  )
}
