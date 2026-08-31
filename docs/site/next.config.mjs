/** @type {import('next').NextConfig} */
const nextConfig = {
  allowedDevOrigins: ["192.168.1.2"],
  experimental: {
    useTypeScriptCli: false,
  },
  output: "export",
  trailingSlash: true,
  poweredByHeader: false,
};

export default nextConfig;
