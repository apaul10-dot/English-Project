const express = require('express');
const router = express.Router();

// Learning route
router.get('/learn', (req, res) => {
  res.render('learn', { 
    title: 'Learn About Food Waste',
    isAuthenticated: req.isAuthenticated(),
    user: req.user,
    active: 'learn'
  });
});

// Food Insecurity route
router.get('/learn/food-insecurity', (req, res) => {
  res.render('learn/food-insecurity', { 
    title: 'Food Insecurity',
    isAuthenticated: req.isAuthenticated(),
    user: req.user,
    active: 'learn'
  });
});

// About route
router.get('/about', (req, res) => {
  res.render('about', { 
    title: 'About Our Research Team',
    isAuthenticated: req.isAuthenticated(),
    user: req.user,
    active: 'about'
  });
});

module.exports = router; 