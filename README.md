# English Project

A web application built with Node.js and Express.

## Features

- Express server with EJS templating
- Multiple routes (/model1, /model2)
- Static file serving
- Ngrok integration for public URL access

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the server:
```bash
PORT=4000 node app.js
```

3. For public URL access (optional):
```bash
ngrok http 4000
```

## Environment Variables

- `PORT`: Server port (default: 3000)
- `NGROK_AUTH_TOKEN`: Your ngrok authentication token

## Project Structure

```
.
├── app.js              # Main application file
├── views/             # EJS templates
├── public/            # Static files
└── package.json       # Project dependencies
```