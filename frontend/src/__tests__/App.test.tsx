import App from "@/App"
import React from "react"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

describe("jest test", () => {
	test('this is a test', () => {
		expect(true).toBe(true)
	})
})

// Mock providers, routes and pages before importing App
jest.mock("@/contexts/auth-context", () => ({
	AuthProvider: ({ children }: { children: React.ReactNode }) => <div data-testid="auth">{children}</div>,
}))

jest.mock("@/components/protected-route", () => ({
	ProtectedRoute: ({ children }: { children: React.ReactNode }) => <div data-testid="protected">{children}</div>,
}))

jest.mock("@/pages/login", () => {
	return {
		__esModule: true,
		default: () => <div>Login Page</div>,
	}
})
jest.mock("@/pages/register", () => {
	return {
		__esModule: true,
		default: () => <div>Register Page</div>,
	}
})
jest.mock("@/pages/dashboard", () => {
	return {
		__esModule: true,
		default: () => <div>Dashboard Page</div>,
	}
})
jest.mock("@/pages/datasets", () => {
	return {
		__esModule: true,
		default: () => <div>Datasets Page</div>,
	}
})
jest.mock("@/pages/models", () => {
	return {
		__esModule: true,
		default: () => <div>Models Page</div>,
	}
})
jest.mock("@/pages/compare", () => {
	return {
		__esModule: true,
		default: () => <div>Compare Page</div>,
	}
})
jest.mock("@/pages/experiments", () => {
	return {
		__esModule: true,
		default: () => <div>Experiments Page</div>,
	}
})


describe("App routing", () => {
	test('We recommend installing an extension to run jest tests.', () => {
		expect(true).toBe(true)
	})

	test("renders LoginPage at /login", () => {
		render(
			<MemoryRouter initialEntries={["/login"]}>
				<App />
			</MemoryRouter>
		)
		expect(screen.getByText("Login Page")).toBeTruthy()
	})

	test("renders RegisterPage at /register", () => {
		render(
			<MemoryRouter initialEntries={["/register"]}>
				<App />
			</MemoryRouter>
		)
		expect(screen.getByText("Register Page")).toBeTruthy()
	})

	test("redirects / to /dashboard and renders DashboardPage", () => {
		render(
			<MemoryRouter initialEntries={["/"]}>
				<App />
			</MemoryRouter>
		)
		expect(screen.getByText("Dashboard Page")).toBeTruthy()
	})

	test("protected routes render their pages when ProtectedRoute allows", () => {
		const protectedPaths = [
			["/datasets", "Datasets Page"],
			["/models", "Models Page"],
			["/compare", "Compare Page"],
			["/experiments", "Experiments Page"],
			["/dashboard", "Dashboard Page"],
		]

		protectedPaths.forEach(([path, text]) => {
			render(
				<MemoryRouter initialEntries={[String(path)]}>
					<App />
				</MemoryRouter>
			)
			expect(screen.getByText(String(text))).toBeTruthy()
		})
	})
})