import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The website reads files from the parent directory, and a package-lock.json
  // in the home directory confuses the workspace detection. This makes the
  // root explicit.
  turbopack: {
    root: path.join(__dirname, ".."),
  },
};

export default nextConfig;
