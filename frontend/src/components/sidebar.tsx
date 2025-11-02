import type React from "react"

import { Database, Settings, FlaskConical, BarChart3, TrendingUp, LogOut, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Link, useLocation } from "react-router-dom"
import { useAuth } from "../contexts/auth-context"

export function Sidebar() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const activePage = location.pathname

  return (
    <aside className="w-60 bg-linear-to-b from-[#6B7FD7] to-[#8B5FBF] text-white flex flex-col">
      {/* Logo */}
      <Link to="/dashboard" className="p-6 flex items-center gap-3">
        <img src="/logo.png" alt="Web ML Logo" className="w-8 h-8" />
        <span className="text-xl font-semibold">Web ML</span>
      </Link>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-1">
        <NavItem
          icon={<BarChart3 className="w-5 h-5" />}
          label="Dashboard"
          active={activePage === "/dashboard" || activePage === "/"}
          to="/dashboard"
        />
        <NavItem
          icon={<Database className="w-5 h-5" />}
          label="Datasets"
          active={activePage === "/datasets"}
          to="/datasets"
        />
        <NavItem
          icon={<Settings className="w-5 h-5" />}
          label="Models"
          active={activePage === "/models"}
          to="/models"
        />
        <NavItem
          icon={<FlaskConical className="w-5 h-5" />}
          label="Experiments"
          active={activePage === "/experiments"}
          to="/experiments"
        />
        <NavItem
          icon={<TrendingUp className="w-5 h-5" />}
          label="Compare Models"
          active={activePage === "/compare"}
          to="/compare"
        />
        <NavItem
          icon={<TrendingUp className="w-5 h-5" />}
          label="Predictions"
          active={activePage === "/predictions"}
          to="/predictions"
        />
      </nav>

      {/* User section */}
      <div className="p-4 space-y-3 sticky bottom-0">
        <div className="flex items-center gap-2 px-3 py-2">
          <User className="w-4 h-4" />
          <span className="text-sm">{user?.email || "Guest"}</span>
        </div>
        <Button
          variant="outline"
          className="w-full bg-transparent border-white/30 text-white hover:bg-white/10 hover:text-white"
          onClick={logout}
        >
          <LogOut className="w-4 h-4 mr-2" />
          Logout
        </Button>
      </div>
    </aside>
  )
}

function NavItem({
  icon,
  label,
  active = false,
  to,
}: { icon: React.ReactNode; label: string; active?: boolean; to: string }) {
  return (
    <Link
      to={to}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
        active ? "bg-white/20" : "hover:bg-white/10"
      }`}
    >
      {icon}
      <span className="text-sm font-medium">{label}</span>
    </Link>
  )
}
