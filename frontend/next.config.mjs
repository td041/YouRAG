/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://backend:8000/:path*',
      },
    ];
  },
};

export default nextConfig;
