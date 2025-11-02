import { Sidebar } from "@/components/sidebar"
import { CompareModels } from "@/components/compare-models"

export default function ComparePage() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1">
        <CompareModels />
      </main>
    </div>
  )
}
