import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Slack Searcher",
  description: "Ask questions about your Slack workspace — answers grounded in real messages",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, background: "#f7f8fa" }}>{children}</body>
    </html>
  );
}
