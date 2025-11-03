import { useRef } from "react";
import html2canvas from "html2canvas";
import { Button } from "@/components/ui/button"; // if using shadcn/ui or similar

export default function DownloadableChart({ children }: { children: React.ReactNode }) {
  const chartRef = useRef<HTMLDivElement>(null);

  const handleDownload = async () => {
    if (!chartRef.current) return;

    const canvas = await html2canvas(chartRef.current, {
      backgroundColor: "#ffffff",
      scale: 2, // for higher quality
    });
    const link = document.createElement("a");
    link.download = "chart.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  return (
    <div className="relative">
      <div ref={chartRef}>{children}</div>
      <Button
        className="absolute top-2 right-2 z-10 text-xs"
        variant="outline"
        onClick={handleDownload}
      >
        ⬇ Download PNG
      </Button>
    </div>
  );
}
