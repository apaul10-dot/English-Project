const express = require('express');
const path = require('path');
const ngrok = require('ngrok');
const axios = require('axios');
const bodyParser = require('body-parser');
const app = express();
const port = process.env.PORT || 3002;


// Set up EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files
app.use(express.static('public'));

// Middleware for parsing form data
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

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

app.post('/model2', async (req, res) => {
    const { household_size, income, meals_per_day, food_expenditure } = req.body;
    try {
        const response = await axios.post('http://localhost:5002/predict', {
            household_size: Number(household_size),
            income: Number(income),
            meals_per_day: Number(meals_per_day),
            food_expenditure: Number(food_expenditure)
        });
        const prediction = response.data.predicted_waste_kg;
        res.render('model2', { prediction });
    } catch (error) {
        res.render('model2', { prediction: 'Error: Could not get prediction.' });
    }
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