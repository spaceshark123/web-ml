import { Sidebar } from "@/components/sidebar"
import { ExperimentsContent } from "@/components/experiments-content"

export default function ComparePage() {
  return (
	<div className="flex min-h-screen">
	  <Sidebar />
	  <main className="flex-1">
		<ExperimentsContent />
	  </main>
	</div>
  )
}
