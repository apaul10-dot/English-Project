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

module.exports = router; 