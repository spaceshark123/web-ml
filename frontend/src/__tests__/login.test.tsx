import React from "react"
import "@testing-library/jest-dom"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import LoginPage from "@/pages/login"

// filepath: /Users/spaceshark/Desktop/web_apps/webml/frontend/src/__tests__/login.test.tsx

// Mocks
const mockLogin = jest.fn()
const mockNavigate = jest.fn()

jest.mock("@/contexts/auth-context", () => ({
	useAuth: () => ({ login: mockLogin }),
}))

jest.mock("react-router-dom", () => {
	// preserve other react-router-dom exports if needed
	const actual = jest.requireActual("react-router-dom")
	return {
		...actual,
		useNavigate: () => mockNavigate,
		Link: ({ children, to, className }: any) => (
			<a href={to} className={className}>
				{children}
			</a>
		),
	}
})

// Simplify UI components to basic HTML elements for testing
jest.mock("@/components/ui/button", () => ({
	Button: ({ children, ...rest }: any) => <button {...rest}>{children}</button>,
}))
jest.mock("@/components/ui/input", () => ({
	Input: (props: any) => <input {...props} />,
}))
jest.mock("@/components/ui/label", () => ({
	Label: (props: any) => <label {...props} />,
}))

// Component under test

describe("LoginPage", () => {
	beforeEach(() => {
		jest.clearAllMocks()
	})

	test("renders form fields and submit button", () => {
		render(<LoginPage />)

		expect(screen.getByRole("heading", { name: /Login to Web ML/i })).toBeInTheDocument()
		expect(screen.getByLabelText(/Email/i)).toBeInTheDocument()
		expect(screen.getByLabelText(/Password/i)).toBeInTheDocument()
		expect(screen.getByRole("button", { name: /Login/i })).toBeInTheDocument()
		// Links present
		expect(screen.getByText(/Forgot your password\?/i)).toBeInTheDocument()
		expect(screen.getByText(/Register here/i)).toBeInTheDocument()
	})

	test("successful login calls login and navigates to dashboard", async () => {
		mockLogin.mockResolvedValueOnce(undefined)

		render(<LoginPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getByLabelText(/Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Login/i })

		await userEvent.type(emailInput, "user@example.com")
		await userEvent.type(passwordInput, "s3cr3t")
		fireEvent.click(submitButton)

		await waitFor(() => {
			expect(mockLogin).toHaveBeenCalledWith("user@example.com", "s3cr3t")
			expect(mockNavigate).toHaveBeenCalledWith("/dashboard")
		})
	})

	test("failed login shows error message returned by login", async () => {
		mockLogin.mockRejectedValueOnce(new Error("Invalid credentials"))

		render(<LoginPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getByLabelText(/Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Login/i })

		await userEvent.type(emailInput, "bad@example.com")
		await userEvent.type(passwordInput, "wrong")
		fireEvent.click(submitButton)

		const errorEl = await screen.findByText(/Invalid credentials/i)
		expect(errorEl).toBeInTheDocument()
		expect(mockNavigate).not.toHaveBeenCalled()
	})

	test("shows fallback error text if rejected value has no message", async () => {
		// reject with a plain value
		// component uses err.message || 'Failed to log in...'
		// jest simulated rejection without message should trigger fallback text
		mockLogin.mockRejectedValueOnce({})

		render(<LoginPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getByLabelText(/Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Login/i })

		await userEvent.type(emailInput, "no-msg@example.com")
		await userEvent.type(passwordInput, "nope")
		fireEvent.click(submitButton)

		const fallback = await screen.findByText(/Failed to log in\. Please check your credentials\./i)
		expect(fallback).toBeInTheDocument()
		expect(mockNavigate).not.toHaveBeenCalled()
	})
})