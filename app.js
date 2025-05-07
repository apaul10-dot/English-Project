const express = require('express');
const path = require('path');
const ngrok = require('ngrok');
const app = express();
const port = process.env.PORT || 3001;


// Set up EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
    res.render('index');
});

app.get('/model1', (req, res) => {
    res.render('model1');
});

app.get('/model2', (req, res) => {
    res.render('model2');
});

app.get('/food-insecurity', (req, res) => {
    res.render('food-insecurity');
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start the server with error handling
const server = app.listen(port, async () => {
    console.log(`Server running at http://localhost:${port}`);
    
    try {
        // You need to set your ngrok authtoken first
        // Run: ngrok authtoken YOUR_AUTH_TOKEN
        const url = await ngrok.connect({
            addr: port,
            authtoken: process.env.NGROK_AUTH_TOKEN // You can set this in your environment
        });
        console.log(`Ngrok tunnel is running at: ${url}`);
    } catch (err) {
        console.error('Error starting ngrok:', err.message);
        console.error('To fix this:');
        console.error('1. Sign up at https://dashboard.ngrok.com/signup');
        console.error('2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken');
        console.error('3. Run: ngrok authtoken YOUR_AUTH_TOKEN');
    }
}).on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`Port ${port} is already in use. Please try a different port.`);
    } else {
        console.error('Server error:', err);
    }
    process.exit(1);
}); 