const express = require('express');
const router = express.Router();

// Games index page
router.get('/', (req, res) => {
    res.render('games/index', {
        title: 'Educational Games',
        active: 'games'
    });
});

// Word Scramble game
router.get('/word-scramble', (req, res) => {
    res.render('games/word-scramble', {
        title: 'Word Scramble Challenge',
        active: 'games'
    });
});

// Memory Match game
router.get('/memory-match', (req, res) => {
    res.render('games/memory-match', {
        title: 'Memory Match Challenge',
        active: 'games'
    });
});

// Food Waste Quiz game
router.get('/quiz', (req, res) => {
    res.render('games/quiz', {
        title: 'Food Waste Quiz',
        active: 'games'
    });
});

// Food Waste Heroes game
router.get('/heroes', (req, res) => {
    res.render('games/heroes', {
        title: 'Food Waste Heroes',
        active: 'games'
    });
});

module.exports = router; 