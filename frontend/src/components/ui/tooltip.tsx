import React, { useRef, useState } from "react"
import { createPortal } from "react-dom"

interface TooltipProps {
  content: string
  children: React.ReactNode
  side?: "top" | "bottom" | "left" | "right"
  offset?: number
}

export function Tooltip({ content, children, side = "top", offset = 8 }: TooltipProps) {
  const ref = useRef<HTMLSpanElement | null>(null)
  const [visible, setVisible] = useState(false)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const [resolvedSide, setResolvedSide] = useState<"top" | "bottom" | "left" | "right">(side)

  const handleEnter = () => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()

    // Auto-flip side if there isn't enough space
    let s = side
    if (side === "top" && rect.top < 40) s = "bottom"
    if (side === "bottom" && window.innerHeight - rect.bottom < 40) s = "top"
    if (side === "left" && rect.left < 40) s = "right"
    if (side === "right" && window.innerWidth - rect.right < 40) s = "left"

    let top = 0
    let left = 0

    if (s === "top") {
      top = rect.top - offset
      left = rect.left + rect.width / 2
    } else if (s === "bottom") {
      top = rect.bottom + offset
      left = rect.left + rect.width / 2
    } else if (s === "left") {
      top = rect.top + rect.height / 2
      left = rect.left - offset
    } else {
      // right
      top = rect.top + rect.height / 2
      left = rect.right + offset
    }

    setResolvedSide(s)
    setPos({ top, left })
    setVisible(true)
  }

  const handleLeave = () => setVisible(false)

  const transform =
    resolvedSide === "top"
      ? "translate(-50%, -100%)"
      : resolvedSide === "bottom"
      ? "translate(-50%, 0)"
      : resolvedSide === "left"
      ? "translate(-100%, -50%)"
      : "translate(0, -50%)" // right

  return (
    <span ref={ref} className="inline-flex items-center" onMouseEnter={handleEnter} onMouseLeave={handleLeave}>
      {children}
      {visible && pos &&
        createPortal(
          <div
            role="tooltip"
            style={{ position: "fixed", top: pos.top, left: pos.left, transform, maxWidth: 320 }}
            className="z-[1000] pointer-events-none rounded bg-black text-white text-xs px-2 py-1 shadow whitespace-normal break-words"
          >
            {content}
          </div>,
          document.body
        )}
    </span>
  )
}
