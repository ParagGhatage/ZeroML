import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Analytics } from "@vercel/analytics/next"
import Footer from "@/components/Footer";
import Navigation from "@/components/Navigation";
import { Toaster } from "@/components/ui/toaster";
import { SpeedInsights } from "@vercel/speed-insights/next"
import { SessionProvider } from "@/context/SessionContext";
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  icons: {
    icon: "/icon.png",
  },
  title: "Zero ML",
  description: "LangFlow for ML",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Navigation/>
       <SessionProvider>
          {children}
        </SessionProvider>
        <SpeedInsights/>
        <Analytics/>
        <Toaster/>
        <Footer />
      </body>
    </html>
  );
}
