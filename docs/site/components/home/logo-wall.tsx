/* eslint-disable @next/next/no-img-element -- static logo files, no optimization pipeline */

type Institution = { name: string; file: string; height: number };

// Where contributors work and study, as listed on their GitHub profiles and in pull requests.
const institutions: Institution[] = [
  { name: "MIT", file: "mit.svg", height: 30 },
  { name: "Stanford University", file: "stanford.svg", height: 30 },
  { name: "Google DeepMind", file: "google-deepmind.svg", height: 28 },
  { name: "Meta", file: "meta.svg", height: 26 },
  { name: "ByteDance Seed", file: "bytedance-seed.svg", height: 26 },
  { name: "UC Berkeley", file: "berkeley.svg", height: 40 },
  { name: "Amazon", file: "amazon.svg", height: 30 },
  { name: "National University of Singapore", file: "nus.png", height: 42 },
  { name: "MiniMax", file: "minimax.png", height: 24 },
  { name: "University of Illinois Urbana-Champaign", file: "uiuc.svg", height: 40 },
  { name: "Northeastern University", file: "northeastern.svg", height: 24 },
  { name: "Carnegie Mellon University", file: "cmu.svg", height: 18 },
];

function Row({ suffix }: { suffix: string }) {
  return (
    <>
      {institutions.map((item) => (
        <span className="logo-cell" key={`${item.file}-${suffix}`} title={item.name}>
          <img src={`/logos/${item.file}`} alt={item.name} style={{ height: item.height }} loading="lazy" decoding="async" />
        </span>
      ))}
    </>
  );
}

// A slow marquee, muted until hovered. The row is rendered twice for a seamless loop.
export function LogoWall() {
  return (
    <div className="logo-wall" role="group" aria-label="Institutions contributors come from">
      <div className="logo-track">
        <Row suffix="a" />
        <Row suffix="b" />
      </div>
    </div>
  );
}
