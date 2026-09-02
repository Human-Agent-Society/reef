import type { Metadata } from "next";
import { Header } from "@/components/header";
import { ThemeSync } from "@/components/theme-sync";
import { getSearchDocuments, navigation } from "@/lib/docs";
import { siteConfig } from "@/lib/site";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: siteConfig.title, template: `%s | Reef Docs` },
  description: siteConfig.description,
  metadataBase: new URL(siteConfig.url),
  openGraph: {
    siteName: "Reef",
    title: siteConfig.title,
    description: siteConfig.description,
    type: "website",
    url: "/",
  },
  twitter: { card: "summary", title: siteConfig.title, description: siteConfig.description },
};

const themeScript = `(function(){try{var t=localStorage.getItem('reef-theme');var m=t==='light'||t==='dark'?t:'auto';var d=m==='dark'||(m==='auto'&&window.matchMedia('(prefers-color-scheme: dark)').matches);var e=document.documentElement;e.dataset.theme=d?'dark':'light';e.dataset.themeMode=m}catch(e){}})()`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const searchDocuments = getSearchDocuments();

  return (
    <html lang="en" data-scroll-behavior="smooth" suppressHydrationWarning>
      <body>
        {/* Inline so it runs at parse time, before first paint. */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <ThemeSync />
        <Header documents={searchDocuments} navigation={navigation} />
        {children}
      </body>
    </html>
  );
}
