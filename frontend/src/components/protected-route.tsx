import type React from "react"

import { Navigate } from "react-router-dom"
import { useAuth } from "../contexts/auth-context"

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
	const { isAuthenticated, isInitialized } = useAuth()
	
	// if auth state is not yet initialized, don't redirect
	if (!isInitialized) {
		return null; // or a loading spinner
	}
	if (!isAuthenticated) {
		return <Navigate to="/login" replace />
	}

	return <>{children}</>
}
