# Use Node 20 LTS image
FROM node:20

# Set working directory
WORKDIR /app

# Copy package.json first (caching layer)
COPY package.json ./

ENV NODE_ENV=development

# Install dependencies
RUN npm install --production=false

# make sure vite is installed by printing its version
RUN npx vite --version

# Copy frontend code
COPY . .

# Expose React dev server port
EXPOSE 5173

ENV VITE_API_URL=http://localhost:5050/api
ENV VITE_SOCKET_URL=http://localhost:5050

# Start the frontend
CMD ["npx", "vite"]