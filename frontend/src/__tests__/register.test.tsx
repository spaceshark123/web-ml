import React from "react"
import "@testing-library/jest-dom"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import RegisterPage from "@/pages/register"

// filepath: /Users/spaceshark/Desktop/web_apps/webml/frontend/src/__tests__/register.test.tsx

// Mocks
const mockRegister = jest.fn()
const mockNavigate = jest.fn()

jest.mock("@/contexts/auth-context", () => ({
	useAuth: () => ({ register: mockRegister }),
}))

jest.mock("react-router-dom", () => {
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


describe("RegisterPage", () => {
	beforeEach(() => {
		jest.clearAllMocks()
	})

	test("renders form fields and submit button", () => {
		render(<RegisterPage />)

		expect(screen.getByRole("heading", { name: /Register for ML Platform/i })).toBeInTheDocument()
		expect(screen.getByLabelText(/Email/i)).toBeInTheDocument()
		expect(screen.getAllByLabelText(/Password/i)).toHaveLength(2)
		expect(screen.getByLabelText(/Confirm Password/i)).toBeInTheDocument()
		expect(screen.getByRole("button", { name: /Register/i })).toBeInTheDocument()
		// Link to login present
		expect(screen.getByText(/Already have an account\?/i)).toBeInTheDocument()
		expect(screen.getByText(/Login here/i)).toBeInTheDocument()
	})

	test("successful register calls register and navigates to dashboard", async () => {
		mockRegister.mockResolvedValueOnce(undefined)

		render(<RegisterPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getAllByLabelText(/Password/i)[0] as HTMLInputElement
		const confirmInput = screen.getByLabelText(/Confirm Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Register/i })

		await userEvent.type(emailInput, "newuser@example.com")
		await userEvent.type(passwordInput, "s3cr3t")
		await userEvent.type(confirmInput, "s3cr3t")
		fireEvent.click(submitButton)

		await waitFor(() => {
			expect(mockRegister).toHaveBeenCalledWith("newuser@example.com", "s3cr3t")
			expect(mockNavigate).toHaveBeenCalledWith("/dashboard")
		})
	})

	test("password mismatch shows error and does not call register or navigate", async () => {
		render(<RegisterPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getAllByLabelText(/Password/i)[0] as HTMLInputElement
		const confirmInput = screen.getByLabelText(/Confirm Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Register/i })

		await userEvent.type(emailInput, "user@example.com")
		await userEvent.type(passwordInput, "password1")
		await userEvent.type(confirmInput, "password2")
		fireEvent.click(submitButton)

		const errorEl = await screen.findByText(/Passwords do not match/i)
		expect(errorEl).toBeInTheDocument()
		expect(mockRegister).not.toHaveBeenCalled()
		expect(mockNavigate).not.toHaveBeenCalled()
	})

	test("failed register shows error message returned by register", async () => {
		mockRegister.mockRejectedValueOnce(new Error("Email already in use"))

		render(<RegisterPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getAllByLabelText(/Password/i)[0] as HTMLInputElement
		const confirmInput = screen.getByLabelText(/Confirm Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Register/i })

		await userEvent.type(emailInput, "taken@example.com")
		await userEvent.type(passwordInput, "mypassword")
		await userEvent.type(confirmInput, "mypassword")
		fireEvent.click(submitButton)

		const errorEl = await screen.findByText(/Email already in use/i)
		expect(errorEl).toBeInTheDocument()
		expect(mockNavigate).not.toHaveBeenCalled()
	})

	test("shows fallback error text if rejected value has no message", async () => {
		mockRegister.mockRejectedValueOnce({})

		render(<RegisterPage />)

		const emailInput = screen.getByLabelText(/Email/i) as HTMLInputElement
		const passwordInput = screen.getAllByLabelText(/Password/i)[0] as HTMLInputElement
		const confirmInput = screen.getByLabelText(/Confirm Password/i) as HTMLInputElement
		const submitButton = screen.getByRole("button", { name: /Register/i })

		await userEvent.type(emailInput, "no-msg@example.com")
		await userEvent.type(passwordInput, "abc123")
		await userEvent.type(confirmInput, "abc123")
		fireEvent.click(submitButton)

		const fallback = await screen.findByText(/Registration failed\. Please try again\./i)
		expect(fallback).toBeInTheDocument()
		expect(mockNavigate).not.toHaveBeenCalled()
	})
})