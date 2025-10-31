import { Sidebar } from "@/components/sidebar"
import { ModelsContent } from "@/components/models-content"

export default function ModelsPage() {
  return (
	<div className="flex min-h-screen">
	  <Sidebar />
	  <main className="flex-1">
		<ModelsContent />
	  </main>
	</div>
  )
}
