import type { Metadata } from "next";
import { DM_Mono, DM_Sans, Source_Serif_4 } from "next/font/google";
import { Header } from "@/components/header";
import { RouteFocus } from "@/components/route-focus";
import { ThemeSync } from "@/components/theme-sync";
import { getSearchDocuments, navigation } from "@/lib/docs";
import { siteConfig } from "@/lib/site";
import "./globals.css";

const dmSans = DM_Sans({
  weight: ["400", "500", "600", "800"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-dm-sans",
});

const sourceSerif = Source_Serif_4({
  // Keep optical sizing available across the site's heading sizes.
  axes: ["opsz"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-source-serif",
});

const dmMono = DM_Mono({
  weight: ["400", "500"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-dm-mono",
});

export const metadata: Metadata = {
  title: { default: siteConfig.title, template: `%s | Reef Docs` },
  description: siteConfig.description,
  metadataBase: new URL(siteConfig.url),
  icons: {
    icon: [
      { url: "/favicon.png", type: "image/png", sizes: "96x96" },
      { url: "/icon.svg", type: "image/svg+xml", sizes: "any" },
    ],
  },
  openGraph: {
    siteName: "Reef",
    title: siteConfig.title,
    description: siteConfig.description,
    type: "website",
    url: "/",
  },
  twitter: { card: "summary", title: siteConfig.title, description: siteConfig.description },
};

const themeScript = `(function(){try{var t=localStorage.getItem('reef-theme');var m=t==='light'||t==='dark'?t:'auto';var d=m==='dark'||(m==='auto'&&window.matchMedia('(prefers-color-scheme: dark)').matches);var e=document.documentElement;e.dataset.theme=d?'dark':'light'}catch(e){}})()`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const searchDocuments = getSearchDocuments();

  return (
    <html lang="en" className={`${dmSans.variable} ${sourceSerif.variable} ${dmMono.variable}`} suppressHydrationWarning>
      <body>
        {/* Inline so it runs at parse time, before first paint. */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <a className="skip-link" href="#main-content">Skip to content</a>
        <ThemeSync />
        <RouteFocus />
        <Header documents={searchDocuments} navigation={navigation} />
        {children}
      </body>
    </html>
  );
}
