import { createContext, useContext, useState, useEffect, type ReactNode } from "react"
import { API_BASE_URL } from "@/constants";
import axios from "axios";

const api = axios.create({
	baseURL: API_BASE_URL,
	withCredentials: true,
});

interface User {
	email: string
}

interface AuthContextType {
	user: User | null
	login: (email: string, password: string) => Promise<boolean>
	register: (email: string, password: string) => Promise<boolean>
	logout: () => void
	isAuthenticated: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
	const [user, setUser] = useState<User | null>(null)

	useEffect(() => {
		// Check if user is already logged in (from localStorage)
		const storedUser = localStorage.getItem("user")
		if (storedUser) {
			setUser(JSON.parse(storedUser))
		}
	}, [])

	const login = async (email: string, password: string): Promise<boolean> => {
		const response = await api.post('/login', { email, password })
		if (response.status === 200) {
			const userData = { email }
			setUser(userData)
			localStorage.setItem("user", JSON.stringify(userData))
			return true
		}
		// error handling
		console.error("Login failed:", response.data)
		return false
	}

	const register = async (email: string, password: string): Promise<boolean> => {
		const response = await api.post('/register', { email, password })
		if (response.status === 201) {
			const userData = { email }
			setUser(userData)
			localStorage.setItem("user", JSON.stringify(userData))
			return true
		}
		// error handling
		console.error("Registration failed:", response.data)
		return false
	}

	const logout = async () => {
		setUser(null)
		await api.post('/logout') // Notify backend about logout
		localStorage.removeItem("user")
	}

	return (
		<AuthContext.Provider value={{ user, login, register, logout, isAuthenticated: !!user }}>
			{children}
		</AuthContext.Provider>
	)
}

export function useAuth() {
	const context = useContext(AuthContext)
	if (context === undefined) {
		throw new Error("useAuth must be used within an AuthProvider")
	}
	return context
}
