import type { Metadata } from "next";
import { Header } from "@/components/header";
import { RouteFocus } from "@/components/route-focus";
import { ThemeSync } from "@/components/theme-sync";
import { getSearchDocuments, navigation } from "@/lib/docs";
import { siteConfig } from "@/lib/site";
import "./globals.css";

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
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Linked here, not imported from the stylesheet, so the browser fetches the fonts in parallel with the CSS. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;800&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" />
      </head>
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
