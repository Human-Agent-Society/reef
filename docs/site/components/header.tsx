import type { NavGroup } from "@/lib/docs";
import { siteConfig } from "@/lib/site";
import { GitHubIcon } from "./github-icon";
import { HeaderNav } from "./header-nav";
import { Logo } from "./logo";
import { Search } from "./search";
import { ThemeToggle } from "./theme-toggle";

type SearchDocument = { slug: string; title: string; description: string; text: string };

export function Header({ documents, navigation }: { documents: SearchDocument[]; navigation: NavGroup[] }) {
  return (
    <header className="site-header">
      <div className="header-inner">
        <Logo />
        <HeaderNav navigation={navigation} />
        <div className="header-search"><Search documents={documents} /></div>
        <nav className="header-actions" aria-label="Global navigation">
          <a className="icon-button" href={siteConfig.repository} target="_blank" rel="noreferrer" aria-label="Reef on GitHub"><GitHubIcon size={19} /></a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
