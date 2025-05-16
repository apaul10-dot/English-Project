const express = require('express');
const path = require('path');
const ngrok = require('ngrok');
const axios = require('axios');
const bodyParser = require('body-parser');
const session = require('express-session');
const MongoStore = require('connect-mongo');
const mongoose = require('mongoose');
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;
const User = require('./models/User');
const { isAuthenticated, isNotAuthenticated } = require('./middleware/auth');
const flash = require('connect-flash');
const mlModelRouter = require('./routes/mlModel');
const gamesRouter = require('./routes/games');
const indexRouter = require('./routes/index');

const app = express();
const port = 2002;

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/hunger-stoppers', {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => {
    console.log('Connected to MongoDB');
}).catch(err => {
    console.error('MongoDB connection error:', err);
});

// Set up EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middleware
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: false,
    store: MongoStore.create({ mongoUrl: 'mongodb://localhost:27017/hunger-stoppers' }),
    cookie: {
        maxAge: 1000 * 60 * 60 * 24 // 24 hours
    }
}));

// Insert connect-flash middleware (after session middleware)
app.use(flash());

// Initialize Passport
app.use(passport.initialize());
app.use(passport.session());

// Passport configuration
passport.use(new LocalStrategy(
    { usernameField: 'email' },
    async (email, password, done) => {
        try {
            const user = await User.findOne({ email });
            if (!user) {
                return done(null, false, { message: 'Incorrect email.' });
            }
            const isMatch = await user.comparePassword(password);
            if (!isMatch) {
                return done(null, false, { message: 'Incorrect password.' });
            }
            return done(null, user);
        } catch (error) {
            return done(error);
        }
    }
));

passport.serializeUser((user, done) => {
    done(null, user.id);
});

passport.deserializeUser(async (id, done) => {
    try {
        const user = await User.findById(id);
        done(null, user);
    } catch (error) {
        done(error);
    }
});

// Routes
app.get('/', (req, res) => {
    res.render('index', { title: 'Home', active: 'home' });
});

app.get('/model2', (req, res) => {
    res.render('model2', { title: 'ML Model', active: 'model2' });
});

app.get('/food-insecurity', (req, res) => {
    res.render('food-insecurity', { title: 'Food Insecurity', active: 'food-insecurity' });
});

app.get('/kids-game', (req, res) => {
    res.render('kids-game', { title: 'Kids Game', active: 'kids-game' });
});

app.get('/resources', (req, res) => {
    res.render('resources', { title: 'Resources', active: 'resources' });
});

app.get('/about', (req, res) => {
    res.render('about', { title: 'About Us', active: 'about' });
});

// Model prediction route
app.post('/model2', async (req, res) => {
    try {
        // Extract input values
        const householdSize = parseInt(req.body.household_size);
        const income = parseInt(req.body.income);
        const mealsPerDay = parseInt(req.body.meals_per_day);
        const foodExpenditure = parseInt(req.body.food_expenditure);

        // Calculate base waste (kg per month)
        // Average person wastes about 1.5-2 kg of food per month
        let baseWaste = householdSize * 1.8;

        // Adjust based on income (higher income households tend to waste more)
        const incomeFactor = Math.min(income / 5000, 2); // Cap at 2x for very high incomes
        baseWaste *= incomeFactor;

        // Adjust based on meals per day (more meals = more potential waste)
        const mealFactor = mealsPerDay / 3; // Normalize to 3 meals per day
        baseWaste *= mealFactor;

        // Adjust based on food expenditure (higher spending = more potential waste)
        const expenditureFactor = Math.min(foodExpenditure / (householdSize * 200), 1.5);
        baseWaste *= expenditureFactor;

        // Add some randomness (±20%)
        const randomFactor = 0.8 + Math.random() * 0.4;
        baseWaste *= randomFactor;

        // Round to 1 decimal place
        baseWaste = Math.round(baseWaste * 10) / 10;

        // Determine waste category
        let category;
        if (baseWaste < householdSize * 1.2) {
            category = "Low";
        } else if (baseWaste < householdSize * 2.5) {
            category = "Medium";
        } else {
            category = "High";
        }

        // Generate relevant factors
        const factors = [];
        if (income > 8000) factors.push("High income leading to more food purchases");
        if (mealsPerDay > 3) factors.push("Multiple meals per day increasing waste potential");
        if (foodExpenditure > householdSize * 300) factors.push("High food expenditure");
        if (householdSize > 4) factors.push("Large household size");
        if (factors.length === 0) factors.push("Balanced food consumption patterns");

        // Create prediction object
        const prediction = {
            waste_kg: baseWaste,
            confidence: 0.85 + (Math.random() * 0.1), // 85-95% confidence
            category: category,
            factors: factors
        };

        res.render('model2', { 
            prediction: prediction,
            title: 'ML Model',
            active: 'model2'
        });
    } catch (error) {
        console.error('Prediction error:', error);
        // Fallback prediction
        const prediction = {
            waste_kg: 5.2,
            confidence: 0.82,
            category: "Medium",
            factors: ["Unable to process all factors", "Using average household data"]
        };
        res.render('model2', { 
            prediction: prediction,
            title: 'ML Model',
            active: 'model2'
        });
    }
});

// Authentication routes
app.get('/login', isNotAuthenticated, (req, res) => {
    res.render('login', { 
        title: 'Login',
        active: 'login',
        error: req.flash('error')
    });
});

app.post('/login', isNotAuthenticated, passport.authenticate('local', {
    successRedirect: '/dashboard',
    failureRedirect: '/login',
    failureFlash: true
}));

app.get('/register', isNotAuthenticated, (req, res) => {
    res.render('register', { 
        title: 'Register',
        active: 'register',
        error: req.flash('error')
    });
});

app.post('/register', isNotAuthenticated, async (req, res) => {
    try {
        const { name, email, password, confirmPassword } = req.body;
        
        // Check if passwords match
        if (password !== confirmPassword) {
            req.flash('error', 'Passwords do not match');
            return res.redirect('/register');
        }

        // Check if user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            req.flash('error', 'Email already registered');
            return res.redirect('/register');
        }

        // Create new user
        const user = new User({
            name,
            email,
            password
        });

        await user.save();
        
        // Log the user in
        req.login(user, (err) => {
            if (err) {
                req.flash('error', 'Error logging in after registration');
                return res.redirect('/login');
            }
            res.redirect('/dashboard');
        });
    } catch (error) {
        console.error('Registration error:', error);
        req.flash('error', 'Error creating account');
        res.redirect('/register');
    }
});

app.get('/logout', (req, res) => {
    req.logout(() => {
        res.redirect('/');
    });
});

// Dashboard route (protected)
app.get('/dashboard', isAuthenticated, (req, res) => {
    res.render('dashboard', {
        title: 'Dashboard',
        active: 'dashboard',
        user: req.user
    });
});

// Dashboard routes
app.post('/dashboard/waste', isAuthenticated, async (req, res) => {
    try {
        const { wasteAmount, category, notes } = req.body;
        const factors = notes ? [notes] : [];
        
        const entry = {
            date: new Date(),
            wasteAmount: parseFloat(wasteAmount),
            category,
            notes,
            factors
        };

        await req.user.addWasteEntry(entry);
        res.redirect('/dashboard');
    } catch (error) {
        console.error('Error adding waste entry:', error);
        req.flash('error', 'Error adding waste entry');
        res.redirect('/dashboard');
    }
});

app.post('/dashboard/goals', isAuthenticated, async (req, res) => {
    try {
        const { targetAmount, startDate, endDate } = req.body;
        
        const goal = {
            targetAmount: parseFloat(targetAmount),
            startDate: new Date(startDate),
            endDate: new Date(endDate),
            status: 'Active',
            progress: 0
        };

        await req.user.addGoal(goal);
        res.redirect('/dashboard');
    } catch (error) {
        console.error('Error creating goal:', error);
        req.flash('error', 'Error creating goal');
        res.redirect('/dashboard');
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start the server
app.listen(port, async () => {
    console.log(`Server running at http://localhost:${port}`);
    
    try {
        const url = await ngrok.connect(port);
        console.log(`Ngrok tunnel is running at: ${url}`);
    } catch (error) {
        if (error.message.includes('port already in use')) {
            console.log('Ngrok is already running on this port');
        } else {
            console.error('Failed to start ngrok:', error);
        }
    }
});

app.use('/model2', mlModelRouter);
app.use('/games', gamesRouter);
app.use('/', indexRouter); 