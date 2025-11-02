import { Sidebar } from "@/components/sidebar"
import { CompareContent } from "@/components/compare-content"

export default function ComparePage() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1">
        <CompareContent />
      </main>
    </div>
  )
}
