import Link from "next/link";
import { Waves } from "lucide-react";

export function Logo() {
  return (
    <Link href="/" className="logo" aria-label="Reef documentation home">
      <Waves className="logo-mark" size={22} strokeWidth={2} aria-hidden="true" />
      <span className="logo-name">REEF</span>
    </Link>
  );
}
