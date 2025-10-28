import type React from "react"

import { useState } from "react"
import { useNavigate, Link } from "react-router-dom"
import { useAuth } from "@/contexts/auth-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export default function LoginPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // TODO: Replace with actual Flask API call
    // Example: const response = await fetch('http://localhost:5000/api/login', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ email, password })
    // })

    const success = await login(email, password)

    if (success) {
      navigate("/dashboard")
    } else {
      alert("Invalid credentials")
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-2xl bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200 px-8 py-6">
          <h1 className="text-4xl font-bold text-gray-900">Login to Web ML</h1>
        </div>

        <form onSubmit={handleSubmit} className="px-8 py-8 space-y-6">
          <div className="space-y-2">
            <Label htmlFor="email" className="text-lg text-gray-700">
              Email
            </Label>
            <Input
              id="email"
              type="text"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="h-12 text-lg border-2 border-gray-900"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password" className="text-lg text-gray-700">
              Password
            </Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="h-12 text-lg border-2 border-gray-300"
              required
            />
          </div>

          <Button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-6 text-lg">
            Login
          </Button>

          <div>
            <Link to="/forgot-password" className="text-blue-600 hover:underline text-lg">
              Forgot your password?
            </Link>
          </div>

          <div className="pt-4">
            <p className="text-gray-600">
              Don't have an account?{" "}
              <Link to="/register" className="text-blue-600 hover:underline">
                Register here
              </Link>
            </p>
          </div>
        </form>
      </div>
    </div>
  )
}
