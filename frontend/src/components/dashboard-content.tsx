import type React from "react"
import { Database, Settings, FlaskConical, TrendingUp, Play, Eye } from "lucide-react"
import { Button } from "./ui/button"
import { Card } from "./ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"

export function DashboardContent() {
  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <h1 className="text-4xl font-bold text-foreground">Dashboard</h1>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={<Database className="w-10 h-10" />}
          value="1"
          label="Datasets"
          gradient="from-[#6B7FD7] to-[#7B6FD7]"
        />
        <StatCard
          icon={<Settings className="w-10 h-10" />}
          value="2"
          label="Models"
          gradient="from-[#7B6FD7] to-[#8B5FBF]"
        />
        <StatCard
          icon={<FlaskConical className="w-10 h-10" />}
          value="2"
          label="Experiments"
          gradient="from-[#8B5FBF] to-[#9B4FAF]"
        />
        <StatCard
          icon={<TrendingUp className="w-10 h-10" />}
          value="0"
          label="Completed"
          gradient="from-[#9B4FAF] to-[#AB3F9F]"
        />
      </div>

      {/* Test Model Performance */}
      <Card className="p-6 space-y-4 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
            <Play className="w-3 h-3 text-white fill-white" />
          </div>
          <h2 className="text-xl font-semibold">Test Model Performance</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Select a Model to Test:</label>
            <div className="flex gap-3">
              <Select>
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Choose a model..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="model1">Model 1</SelectItem>
                  <SelectItem value="model2">Model 2</SelectItem>
                </SelectContent>
              </Select>
              <Button className="bg-green-600 hover:bg-green-700 text-white">
                <Play className="w-4 h-4 mr-2" />
                Run Performance Test
              </Button>
              <Button variant="outline" className="border-info text-info hover:bg-info/10 bg-transparent">
                <Eye className="w-4 h-4 mr-2" />
                View Details
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Datasets */}
        <Card className="p-6 space-y-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-primary" />
              <h2 className="text-xl font-semibold">Recent Datasets</h2>
            </div>
            <Button size="sm" className="bg-blue-600 hover:bg-blue-700 text-white">
              + Upload
            </Button>
          </div>

          <div className="space-y-4">
            <div className="flex items-start justify-between py-3 border-b">
              <div>
                <h3 className="font-semibold">titanic</h3>
                <p className="text-sm text-muted-foreground">891 rows, 12 columns</p>
              </div>
              <span className="text-sm text-muted-foreground">1 month ago</span>
            </div>
          </div>

          <Button variant="outline" className="w-full border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent">
            View All Datasets
          </Button>
        </Card>

        {/* Recent Experiments */}
        <Card className="p-6 space-y-4 shadow-sm">
          <div className="flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-green-600" />
            <h2 className="text-xl font-semibold">Recent Experiments</h2>
          </div>

          <div className="space-y-3">
            <ExperimentItem
              title="Quick Training: titanic on titanic"
              dataset="titanic"
              status="Running"
              time="2 weeks, 1 day ago"
            />
            <ExperimentItem
              title="Quick Training: l on titanic"
              dataset="l"
              status="Running"
              time="2 weeks, 1 day ago"
            />
          </div>

          <Button variant="outline" className="w-full border-blue-600 text-blue-600 hover:bg-blue-50 bg-transparent">
            View All Experiments
          </Button>
        </Card>
      </div>
    </div>
  )
}

function StatCard({
  icon,
  value,
  label,
  gradient,
}: { icon: React.ReactNode; value: string; label: string; gradient: string }) {
  return (
    <Card className={`p-6 bg-gradient-to-br ${gradient} text-white shadow-lg border-0`}>
      <div className="flex flex-col items-center text-center space-y-3">
        {icon}
        <div className="text-4xl font-bold">{value}</div>
        <div className="text-sm font-medium opacity-90">{label}</div>
      </div>
    </Card>
  )
}

function ExperimentItem({
  title,
  dataset,
  status,
  time,
}: { title: string; dataset: string; status: string; time: string }) {
  return (
    <div className="flex items-start justify-between py-3 border-b last:border-0">
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="font-semibold text-sm">{title}</h3>
          <span className="px-2 py-0.5 bg-gray-600 text-white text-xs rounded">{status}</span>
        </div>
        <p className="text-sm text-muted-foreground">{dataset}</p>
      </div>
      <span className="text-sm text-muted-foreground whitespace-nowrap ml-4">{time}</span>
    </div>
  )
}
