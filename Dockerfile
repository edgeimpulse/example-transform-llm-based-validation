FROM node:20.13.1

WORKDIR /app

# Dependencies
COPY package*.json ./
RUN npm ci

# Build the application
COPY . ./
RUN npm run build

ENTRYPOINT [ "node",  "build/llm-validation.js" ]