import { createContext, useContext, useState, useEffect, type ReactNode } from "react"
import { API_BASE_URL } from "@/constants";
import axios from "axios";

const api = axios.create({
	baseURL: API_BASE_URL,
	withCredentials: true,
	headers: {
		'Content-Type': 'application/json',
		'Accept': 'application/json',
	},
});

// Add request interceptor to handle errors
api.interceptors.response.use(
	(response) => response,
	(error) => {
		if (error.response?.status === 401) {
			localStorage.removeItem("user");
		}
		return Promise.reject(error);
	}
);

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
		try {
			const response = await api.post('/login', { email, password });
			const userData = response.data.user;
			setUser(userData);
			localStorage.setItem("user", JSON.stringify(userData));
			return true;
		} catch (error: any) {
			console.error("Login failed:", error);
			// Throw the error with the backend message if available
			throw new Error(error.response?.data?.error || 'Login failed. Please try again.');
		}
	}

	const register = async (email: string, password: string): Promise<boolean> => {
		try {
			const response = await api.post('/register', { email, password });
			if (response.data.error) {
				throw new Error(response.data.error);
			}
			// After successful registration, log in automatically
			return await login(email, password);
		} catch (error: any) {
			console.error("Registration failed:", error);
			throw new Error(error.response?.data?.error || 'Registration failed. Please try again.');
		}
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
